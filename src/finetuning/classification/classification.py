import json
import sqlite3
from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import random
from tqdm import tqdm
import time
import math
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score

root_dir = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi")
sys.path.append(str(root_dir.resolve()))  
sys.path.append(str((root_dir/"src").resolve()))  
sys.path.append(str((root_dir / "chronos_pkg/src").resolve()))

from chronos_pkg.src.chronos import ChronosPipeline


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#adapt db path
DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/classification/classification_cv_new.db"
SEED = 42

#custom cv splits due to small datasets
def create_cv_splits(y, n_splits=5, seed=42):
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    fold_val_indices = [[] for _ in range(n_splits)]

    for cls in np.unique(y):
        indices = np.where(y == cls)[0]
        rng.shuffle(indices)

        if len(indices) == 1:
            continue

        for i, idx in enumerate(indices):
            fold_val_indices[i % n_splits].append(idx)

    all_indices = np.arange(len(y))

    for fold in range(n_splits):
        val_idx = np.array(fold_val_indices[fold], dtype=int)
        train_idx = np.setdiff1d(all_indices, val_idx)

        yield train_idx, val_idx

def get_db_connection():
    conn = sqlite3.connect(
        DB_PATH,
        timeout=60,
    )
    conn.execute("PRAGMA busy_timeout=60000")
    return conn

#rerun db on fail
def execute_db_update(sql, params, description="database update"):
    for attempt in range(5):

        conn = None

        try:
            conn = get_db_connection()

            conn.execute(sql, params)
            conn.commit()
            conn.close()

            return

        except sqlite3.OperationalError as e:

            if conn is not None:
                conn.close()

            if "database is locked" not in str(e):
                raise

            print(
                f"[SQLite] Database locked during {description} "
                f"- retry {attempt + 1}/5",
                flush=True
            )

            if attempt < 4:
                time.sleep(5)

    raise RuntimeError(
        f"[SQLite] Database remained locked during "
        f"{description} after 5 attempts."
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_ucr_tsv(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t", header=None).values

    y = df[:, 0]
    X = df[:, 1:].astype(np.float32)

    y = y.astype(int)
    unique = np.unique(y)
    label_map = {v: i for i, v in enumerate(unique)}
    y = np.vectorize(label_map.get)(y)

    return X, y


class ClassificationDataset(Dataset):

    def __init__(self, X, y, tokenizer):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        series = torch.tensor(self.X[idx], dtype=torch.float32)

        token_ids, attention_mask, _ = self.tokenizer.context_input_transform(
            series.unsqueeze(0)
        )

        return {
            "input_ids": token_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": torch.tensor(self.y[idx], dtype=torch.long),
        }

def evaluate_model(model, tokenizer, X, y, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    preds_all, labels_all = [], []

    with torch.no_grad():
        for i in tqdm(range(0, len(X), batch_size), desc="Evaluating"):
            batch_X = X[i:i + batch_size]
            batch_y = y[i:i + batch_size]

            context = torch.tensor(batch_X, dtype=torch.float32)
            

            input_ids, attention_mask, _ = tokenizer.context_input_transform(context)

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs["logits"]

            if logits.ndim == 3:
                logits = logits[:, -1, :]

            preds = torch.argmax(logits, dim=-1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(batch_y)

    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average="weighted")


    return {
        "accuracy": acc,
        "f1": f1,
        "predictions": preds_all,
        "labels": labels_all,
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()

    idx = args.index

    set_seed(SEED)

    conn = sqlite3.connect(
        DB_PATH,
        timeout=120
    )

    conn.execute("PRAGMA busy_timeout=120000")
    cur = conn.cursor()

    cur.execute(
        """
        SELECT config, train_data, dataset
        FROM runs
        WHERE id=?
        """,
        (idx,),
    )

    config_json, train_data, dataset = cur.fetchone()

    conn.close()

    config = json.loads(config_json)

    #Load HPs
    num_labels = config["num_labels"]
    num_epochs = config["num_train_epochs"]
    batch_size = config["per_device_train_batch_size"]
    learning_rate = config["learning_rate"]
    dropout_head = config["dropout_head"]
    warmup_ratio = config["warmup_ratio"]
    train_inner = config["TrainInnerModel"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]



    #Select Model
    pipeline = ChronosPipeline.from_pretrained(
        "ddrg/ChronosBERT-Default",
        task="classification",
        num_labels=num_labels,
        dropout_head=dropout_head,
        TrainInnerModel=train_inner,
    )

    model = pipeline.model.to(DEVICE)
    tokenizer = pipeline.tokenizer

    # load all data
    X, y = load_ucr_tsv(train_data)

    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(
        create_cv_splits(y, n_splits=5, seed=SEED),
        start=1
    ):

        print(f"Fold {fold}/5")

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        set_seed(SEED + fold)

        pipeline = ChronosPipeline.from_pretrained(
            "juliushanusch/ChronosBERT-Default",
            task="classification",
            num_labels=num_labels,
            dropout_head=dropout_head,
            TrainInnerModel=train_inner,
        )

        model = pipeline.model.to(DEVICE)
        tokenizer = pipeline.tokenizer

        train_dataset = ClassificationDataset(X_train, y_train, tokenizer)

        generator = torch.Generator()
        generator.manual_seed(SEED + fold)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
        )

        total_steps = math.ceil(
            num_epochs * len(train_loader) / gradient_accumulation_steps
        )

        warmup_steps = int(total_steps * warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        model.train()
        optimizer.zero_grad()

        for epoch in range(num_epochs):

            total_loss = 0.0

            for step, batch in enumerate(train_loader):

                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs["loss"]
                loss = loss / gradient_accumulation_steps
                loss.backward()

                if (
                    (step + 1) % gradient_accumulation_steps == 0
                    or (step + 1) == len(train_loader)
                ):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * gradient_accumulation_steps

            print(
                f"Fold {fold}, Epoch {epoch + 1}: "
                f"{total_loss / len(train_loader):.4f}"
            )

        results = evaluate_model(
            model,
            tokenizer,
            X_val,
            y_val,
            batch_size
        )

        cv_results.append(results)

        print(
            f"Fold {fold}: "
            f"Accuracy={results['accuracy']:.4f}, "
            f"F1={results['f1']:.4f}"
        )

        del model
        del pipeline
        del tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cv_accuracies = [
        result["accuracy"]
        for result in cv_results
        if not np.isnan(result["accuracy"])
    ]

    cv_f1s = [
        result["f1"]
        for result in cv_results
        if not np.isnan(result["f1"])
    ]

    cv_accuracy = np.mean(cv_accuracies) if cv_accuracies else np.nan
    cv_f1 = np.mean(cv_f1s) if cv_f1s else np.nan

    print(
        f"CV Accuracy: {cv_accuracy:.4f}"
    )

    print(
        f"CV F1: {cv_f1:.4f}"
    )

    execute_db_update(
        """
        UPDATE runs
        SET accuracy = ?
        WHERE id = ?
        """,
        (
            cv_accuracy,
            idx,
        ),
        description="CV Accuracy"
    )
            
