import json
import sqlite3
from pathlib import Path
import sys
import argparse
import numpy as np
import math
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import torch
import random
import time
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.nn.utils.rnn import pad_sequence


root_dir = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi")

sys.path.append(str(root_dir.resolve()))
sys.path.append(str((root_dir/"src").resolve()))
sys.path.append(str((root_dir/"chronos_pkg/src").resolve()))

from chronos_pkg.src.chronos import ChronosPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/similarity/similarity_cv.db"

SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_db_connection():
    conn = sqlite3.connect(
        DB_PATH,
        timeout=60,
    )
    conn.execute("PRAGMA busy_timeout=60000")
    return conn

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


def load_pairs(path):

    data = np.load(
        path,
        allow_pickle=True
    )

    return (
        data["pairs_1"],
        data["pairs_2"],
        data["labels"]
    )

def similarity_collate(batch):

    return {
        "input_ids_1": pad_sequence(
            [b["input_ids_1"] for b in batch],
            batch_first=True,
            padding_value=0
        ),
        "attention_mask_1": pad_sequence(
            [b["attention_mask_1"] for b in batch],
            batch_first=True,
            padding_value=0
        ),
        "input_ids_2": pad_sequence(
            [b["input_ids_2"] for b in batch],
            batch_first=True,
            padding_value=0
        ),
        "attention_mask_2": pad_sequence(
            [b["attention_mask_2"] for b in batch],
            batch_first=True,
            padding_value=0
        ),
        "labels": torch.stack(
            [b["labels"] for b in batch]
        )
    }

class SimilarityDataset(Dataset):
    def __init__(self, X1, X2, y, tokenizer):
        self.X1 = X1
        self.X2 = X2
        self.y = y
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.y)
    
    def encode(self,x):
        ids,mask,_=self.tokenizer.context_input_transform(
            torch.tensor(x).unsqueeze(0)
        )
        return ids.squeeze(0),mask.squeeze(0)
    
    def __getitem__(self,i):
        ids1,mask1=self.encode(self.X1[i])
        ids2,mask2=self.encode(self.X2[i])

        return {
            "input_ids_1":ids1,
            "attention_mask_1":mask1,
            "input_ids_2":ids2,
            "attention_mask_2":mask2,
            "labels":torch.tensor(self.y[i],dtype=torch.float)
        }


@torch.no_grad()
def evaluate_similarity(
    model,
    tokenizer,
    dataset_name,
    X1,
    X2,
    y,
    batch_size=32
):

    device = next(model.parameters()).device
    similarities = []
    model.eval()

    for start in tqdm(
        range(0, len(y), batch_size),
        desc="Evaluating similarity",
        total=(len(y) + batch_size - 1) // batch_size
    ):

        end = start + batch_size

        ids1_list = []
        mask1_list = []
        ids2_list = []
        mask2_list = []

        for a, b in zip(
            X1[start:end],
            X2[start:end]
        ):

            a = np.asarray(a, dtype=np.float32)
            b = np.asarray(b, dtype=np.float32)

            i1, m1, _ = tokenizer.context_input_transform(
                torch.from_numpy(a).unsqueeze(0)
            )

            i2, m2, _ = tokenizer.context_input_transform(
                torch.from_numpy(b).unsqueeze(0)
            )

            ids1_list.append(i1.squeeze(0))
            mask1_list.append(m1.squeeze(0))

            ids2_list.append(i2.squeeze(0))
            mask2_list.append(m2.squeeze(0))

        ids1 = pad_sequence(
            ids1_list,
            batch_first=True,
            padding_value=tokenizer.config.pad_token_id
        ).to(device)

        mask1 = pad_sequence(
            mask1_list,
            batch_first=True,
            padding_value=0
        ).to(device)

        ids2 = pad_sequence(
            ids2_list,
            batch_first=True,
            padding_value=tokenizer.config.pad_token_id
        ).to(device)

        mask2 = pad_sequence(
            mask2_list,
            batch_first=True,
            padding_value=0
        ).to(device)

        outputs = model(
            input_ids_1=ids1,
            attention_mask_1=mask1,
            input_ids_2=ids2,
            attention_mask_2=mask2,
        )

        z1 = outputs["embeddings_1"]
        z2 = outputs["embeddings_2"]

        # cosine similarity [-1,1]
        sim = F.cosine_similarity(
            z1,
            z2,
            dim=-1
        )

        similarities.append(
            sim.cpu()
        )

    similarities = torch.cat(
        similarities
    ).numpy()


    probs = (similarities + 1) / 2
    preds = (
        probs > 0.5
    ).astype(int)

    return {
        "accuracy": accuracy_score(
            y,
            preds
        ),
        "f1": f1_score(
            y,
            preds
        ),
        "auroc": roc_auc_score(
            y,
            probs
        )
    }




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Batch index. Each index runs 100 configurations."
    )
    args = parser.parse_args()

    batch_index = args.index

    CONFIGS_PER_INDEX = 100

    start_id = (batch_index - 1) * CONFIGS_PER_INDEX + 1
    end_id = batch_index * CONFIGS_PER_INDEX


    set_seed(SEED)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, config, train_data, dataset, task, accuracy
        FROM runs
        WHERE id >= ?
          AND id <= ?
        ORDER BY id
        """,
        (start_id, end_id)
    )

    runs = cur.fetchall()
    conn.close()

    print(f"Found {len(runs)} configurations.")

    if not runs:
        print("No configurations found.")
        sys.exit(0)


    for run_number, (
        idx,
        config_json,
        train_data,
        dataset_name,
        task,
        existing_accuracy
    ) in enumerate(runs, start=1):


        if existing_accuracy is not None:
            print(
                f"Skipping configuration {run_number}/{len(runs)} "
                f"(DB id={idx}) - accuracy already exists: "
                f"{existing_accuracy:.4f}"
            )
            continue

        print()
        print("#" * 80)
        print(
            f"Configuration {run_number}/{len(runs)} "
            f"(DB id={idx})"
        )
        print(f"Dataset: {dataset_name}")
        print("#" * 80)

        config = json.loads(config_json)

        num_epochs = config["num_train_epochs"]
        batch_size = config["per_device_train_batch_size"]
        gradient_accumulation_steps = config[
            "gradient_accumulation_steps"
        ]
        learning_rate = config["learning_rate"]
        dropout_head = config["dropout_head"]
        warmup_ratio = config["warmup_ratio"]
        train_inner = config["TrainInnerModel"]


        X1, X2, labels = load_pairs(train_data)

        total_nan_x1 = 0
        total_nan_x2 = 0
        total_values_x1 = 0
        total_values_x2 = 0

        for a, b in zip(X1, X2):
            a = np.asarray(a, dtype=np.float32)
            b = np.asarray(b, dtype=np.float32)

            total_nan_x1 += np.isnan(a).sum()
            total_nan_x2 += np.isnan(b).sum()
            total_values_x1 += a.size
            total_values_x2 += b.size



        invalid_pairs = []

        for i, (a, b) in enumerate(zip(X1, X2)):
            a = np.asarray(a, dtype=np.float32)
            b = np.asarray(b, dtype=np.float32)

            if np.isnan(a).all() or np.isnan(b).all():
                invalid_pairs.append(i)


        cv_results = []


        for fold, (train_idx, val_idx) in enumerate(
            create_cv_splits(
                labels,
                n_splits=5,
                seed=SEED
            ),
            start=1
        ):

            print(f"Fold {fold}/5")

            X1_train = X1[train_idx]
            X2_train = X2[train_idx]
            y_train = labels[train_idx]

            X1_val = X1[val_idx]
            X2_val = X2[val_idx]
            y_val = labels[val_idx]

            set_seed(SEED + fold)


            pipeline = ChronosPipeline.from_pretrained(
                "ddrg/ChronosBERT-Default",
                task="similarity",
                TrainInnerModel=train_inner,
            )

            model = pipeline.model.to(DEVICE)
            tokenizer = pipeline.tokenizer

            a = np.asarray(X1[533], dtype=np.float32)
            b = np.asarray(X2[533], dtype=np.float32)

            i1, m1, _ = tokenizer.context_input_transform(
                torch.from_numpy(a).unsqueeze(0)
            )

            i2, m2, _ = tokenizer.context_input_transform(
                torch.from_numpy(b).unsqueeze(0)
            )


            dataset = SimilarityDataset(
                X1_train,
                X2_train,
                y_train,
                tokenizer
            )

            generator = torch.Generator()
            generator.manual_seed(SEED + fold)

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=similarity_collate,
                generator=generator
            )

            optimizer = AdamW(
                filter(
                    lambda p: p.requires_grad,
                    model.parameters()
                ),
                lr=learning_rate
            )

            steps_per_epoch = math.ceil(
                len(loader) /
                gradient_accumulation_steps
            )

            total_steps = (
                steps_per_epoch *
                num_epochs
            )

            warmup_steps = int(
                total_steps *
                warmup_ratio
            )

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )


            model.train()
            optimizer.zero_grad()

            for epoch in range(num_epochs):

                loss_total = 0.0

                for step, batch in enumerate(loader):

                    out = model(
                        input_ids_1=batch[
                            "input_ids_1"
                        ].to(DEVICE),

                        attention_mask_1=batch[
                            "attention_mask_1"
                        ].to(DEVICE),

                        input_ids_2=batch[
                            "input_ids_2"
                        ].to(DEVICE),

                        attention_mask_2=batch[
                            "attention_mask_2"
                        ].to(DEVICE),

                        labels=batch[
                            "labels"
                        ].to(DEVICE),
                    )

                    loss = out["loss"]

                    loss_total += loss.item()

                    loss = (
                        loss /
                        gradient_accumulation_steps
                    )

                    loss.backward()

                    if (
                        (step + 1) %
                        gradient_accumulation_steps == 0
                        or
                        (step + 1) ==
                        len(loader)
                    ):

                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                print(
                    f"Fold {fold}, "
                    f"Epoch {epoch + 1}: "
                    f"{loss_total / len(loader):.4f}"
                )


            results = evaluate_similarity(
                model,
                tokenizer,
                dataset_name,
                X1_val,
                X2_val,
                y_val,
                batch_size
            )

            cv_results.append(results)

            print(
                f"Fold {fold}: "
                f"Accuracy={results['accuracy']:.4f}, "
                f"F1={results['f1']:.4f}, "
                f"AUROC={results['auroc']:.4f}"
            )


            del model
            del pipeline
            del tokenizer

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


        cv_accuracy = np.mean([
            result["accuracy"]
            for result in cv_results
        ])

        cv_f1 = np.mean([
            result["f1"]
            for result in cv_results
        ])

        cv_auroc = np.mean([
            result["auroc"]
            for result in cv_results
        ])

        print(
            f"CV Average: "
            f"Accuracy={cv_accuracy:.4f}, "
            f"F1={cv_f1:.4f}, "
            f"AUROC={cv_auroc:.4f}"
        )


        execute_db_update(
            """
            UPDATE runs
            SET accuracy=?,
                f1=?,
                auroc=?
            WHERE id=?
            """,
            (
                cv_accuracy,
                cv_f1,
                cv_auroc,
                idx
            ),
            description=f"CV results for run {idx}"
        )

        print(
            f"Finished configuration {idx}."
        )