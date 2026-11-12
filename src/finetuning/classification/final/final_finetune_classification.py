import json
import sqlite3
from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

root_dir = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi")
sys.path.append(str(root_dir.resolve()))  
sys.path.append(str((root_dir/"src").resolve()))  
sys.path.append(str((root_dir / "chronos_pkg/src").resolve()))

from chronos_pkg.src.chronos import ChronosPipeline


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/classification/final/classification_cv_best.db"


def load_ucr_tsv(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t", header=None).values

    y = df[:, 0]
    X = df[:, 1:].astype(np.float32)

    y = y.astype(int)
    unique = np.unique(y)
    label_map = {v: i for i, v in enumerate(unique)}
    y = np.vectorize(label_map.get)(y)

    return X, y


def load_uci_har(test_dir):

    test_dir = Path(test_dir)

    x_path = test_dir / "X_train.txt"
    y_path = test_dir / "y_train.txt"

    X = np.loadtxt(x_path).astype(np.float32)
    y = np.loadtxt(y_path).astype(int) - 1

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

def get_num_labels(train_file):
    """
    Determine number of classes from the first column
    of the UCR TRAIN_small file.
    """

    labels = set()

    with open(train_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            label = line.split("\t")[0]
            labels.add(label)

    return len(labels)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    idx = args.index
    seed = args.seed

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    conn = sqlite3.connect(
        DB_PATH,
        timeout=120
    )

    conn.execute("PRAGMA busy_timeout=120000")
    cur = conn.cursor()

    model_path_column = f"model_path_{seed}"

    cur.execute(
        f"""
        SELECT config, train_data, dataset, {model_path_column}
        FROM runs
        WHERE id=?
        """,
        (idx,),
    )

    config_json, train_data, dataset, model_path = cur.fetchone()

    output_dir = Path(model_path)

    config = json.loads(config_json)

    #Load HPs
    num_labels = config["num_labels"]

    print(num_labels)

    new_labels =  get_num_labels(train_data)

    
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


    if dataset == "UCI-HAR":
        X, y = load_uci_har(train_data)
    else:
        X, y = load_ucr_tsv(train_data)

    train_dataset = ClassificationDataset(X, y, tokenizer)

    generator = torch.Generator()
    generator.manual_seed(seed)

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
    import math

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

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()

            # Update only after accumulating gradients
            if (
                (step + 1) % gradient_accumulation_steps == 0
                or (step + 1) == len(train_loader)
            ):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps

        print(
            f"Epoch {epoch + 1}: "
            f"{total_loss / len(train_loader):.4f}"
        )

    save_path = output_dir
    save_path.mkdir(parents=True, exist_ok=True)


    model.model.save_pretrained(save_path)

    torch.save(
        model.classifier.state_dict(),
        save_path / "classifier.pt"
    )

    with open(save_path / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)


    print(f"Saved checkpoint: {save_path}")