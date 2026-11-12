import json
import sqlite3
from pathlib import Path
import sys
import argparse
import numpy as np
import torch
import math
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random

from gluonts.dataset.arrow import ArrowFile


root_dir = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi")
sys.path.append(str(root_dir.resolve()))
sys.path.append(str((root_dir/"src").resolve()))
sys.path.append(str((root_dir / "chronos_pkg/src").resolve()))

from chronos_pkg.src.chronos import ChronosPipeline


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/tser/final/tser_RMSE.db"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_arrow(path: Path):
    """
    Reads TS-ARROW dataset using GluonTS ArrowFile reader.
    """

    dataset = ArrowFile(path)

    series = []
    labels = []

    for entry in dataset:
        target = np.asarray(entry["target"], dtype=np.float32)

        if "label" in entry:
            label = entry["label"]
        elif "y" in entry:
            label = entry["y"]
        else:
            raise KeyError("No label found in dataset entry")

        series.append(target)
        labels.append(float(label))

    return np.stack(series), np.array(labels, dtype=np.float32)


class TSERDataset(Dataset):

    def __init__(self, X, y, tokenizer):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        series = torch.tensor(
            self.X[idx],
            dtype=torch.float32
        )

        token_ids, attention_mask, _ = self.tokenizer.context_input_transform(
            series.unsqueeze(0)
        )

        return {
            "input_ids": token_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": torch.tensor(
                self.y[idx],
                dtype=torch.float32
            ),
        }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)

    args = parser.parse_args()

    idx = args.index
    seed = args.seed

    set_seed(seed)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    model_path_column = f"model_path_{seed}"

    cur.execute(
        f"""
        SELECT config,
            train_data,
            dataset,
            {model_path_column}
        FROM runs
        WHERE id=?
        """,
        (idx,),
    )

    row = cur.fetchone()

    conn.close()
    config_json, train_data, dataset, model_path = row
    config = json.loads(config_json)

    output_dir = Path(model_path)

    # Hyperparameters
    batch_size = config["per_device_train_batch_size"]
    num_epochs = config["num_train_epochs"]
    learning_rate = config["learning_rate"]
    dropout_head = config["dropout_head"]
    warmup_ratio = config["warmup_ratio"]
    train_inner = config["TrainInnerModel"]
    loss_type = config["loss_type"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]


    # Load TSER model
    pipeline = ChronosPipeline.from_pretrained(
        "ddrg/ChronosBERT-Default",
        task="tser",
        loss_type=loss_type,
        dropout_head=dropout_head,
        TrainInnerModel=train_inner,
    )


    model = pipeline.model.to(DEVICE)
    tokenizer = pipeline.tokenizer


    # Load data
    X, y = load_arrow(
        Path(train_data)
    )


    dataset = TSERDataset(
        X,
        y,
        tokenizer
    )


    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )


    optimizer = AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=learning_rate,
    )


    steps_per_epoch = math.ceil(
        len(loader) / gradient_accumulation_steps
    )

    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(
        total_steps * warmup_ratio
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.train()
    optimizer.zero_grad()

    for epoch in range(num_epochs):

        total_loss = 0.0

        for step, batch in enumerate(loader):

            outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
            )

            loss = outputs["loss"]

            # Keep the unscaled loss for logging
            total_loss += loss.item()

            # Scale loss before accumulating gradients
            loss = loss / gradient_accumulation_steps
            loss.backward()

            # Update weights only after accumulating gradients
            if (
                (step + 1) % gradient_accumulation_steps == 0
                or (step + 1) == len(loader)
            ):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        print(
            f"Epoch {epoch + 1}: "
            f"{total_loss / len(loader):.4f}"
        )


    # Save
    save_path = output_dir
    save_path.mkdir(
        parents=True,
        exist_ok=True
    )


    model.model.save_pretrained(save_path)


    torch.save(
        model.regressor.state_dict(),
        save_path / "regressor.pt"
    )


    with open(save_path / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)


    print(
        f"Saved checkpoint: {save_path}"
    )