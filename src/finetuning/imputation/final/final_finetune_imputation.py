import json
import sqlite3
from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


root_dir = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi"
)

sys.path.append(str(root_dir.resolve()))
sys.path.append(str((root_dir / "src").resolve()))
sys.path.append(str((root_dir / "chronos_pkg/src").resolve()))

from chronos_pkg.src.chronos import ChronosPipeline


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DB_PATH = (
    "/data/horse/ws/juha972b-AION-BERT-Chronos/"
    "BERTi/src/finetuning/imputation/final/imputation_defaultConfigs.db"
)

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


class ImputationDataset(Dataset):

    def __init__(
        self,
        npz_path,
        tokenizer,
        masking_prob=0.3,
        mean_span_length=16,
    ):
        data = np.load(
            npz_path,
            allow_pickle=True,
        )

        self.series = data["series"]
        self.tokenizer = tokenizer
        self.masking_prob = masking_prob
        self.mean_span_length = mean_span_length

        # Ensure a mask token exists
        if not hasattr(self.tokenizer.config, "mask_token_id"):
            self.tokenizer.config.mask_token_id = (
                self.tokenizer.config.n_special_tokens
            )

            self.tokenizer.config.n_special_tokens += 1
            self.tokenizer.config.n_tokens += 1

            print(
                f"Added <mask> token with id "
                f"{self.tokenizer.config.mask_token_id}"
            )

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):

        series = np.asarray(
            self.series[idx],
            dtype=np.float32,
        )

        # Remove NaN padding if present
        if np.isnan(series).any():
            series = series[~np.isnan(series)]

        series = torch.tensor(
            series,
            dtype=torch.float32,
        )

        input_ids, attention_mask, scale = (
            self.tokenizer.context_input_transform(
                series.unsqueeze(0)
            )
        )

        labels = input_ids.clone()

        mask_token_id = self.tokenizer.config.mask_token_id

        special_tokens_mask = (
            input_ids < self.tokenizer.config.n_special_tokens
        )

        valid_positions = ~special_tokens_mask

        n_tokens = valid_positions.sum().item()

        n_to_mask = max(
            1,
            int(self.masking_prob * n_tokens),
        )

        mask = torch.zeros_like(
            input_ids,
            dtype=torch.bool,
        )

        valid_indices = valid_positions.nonzero(
            as_tuple=True
        )[1]

        total_masked = 0

        while total_masked < n_to_mask:

            span_len = max(
                1,
                int(
                    torch.poisson(
                        torch.tensor(
                            float(self.mean_span_length)
                        )
                    ).item()
                ),
            )

            if total_masked + span_len > n_to_mask:
                span_len = n_to_mask - total_masked

            start = valid_indices[
                torch.randint(
                    len(valid_indices),
                    (1,),
                )
            ].item()

            end = min(
                start + span_len,
                input_ids.size(1),
            )

            if mask[0, start:end].any():
                continue

            mask[0, start:end] = True
            total_masked += span_len

        mask &= valid_positions

        random_prob = torch.rand_like(
            input_ids.float()
        )

        # 80% -> MASK token
        input_ids[
            mask & (random_prob < 0.8)
        ] = mask_token_id

        # 10% -> random nearby token
        random_mask = (
            mask
            & (random_prob >= 0.8)
            & (random_prob < 0.9)
        )

        if random_mask.any():

            indices = torch.nonzero(
                random_mask,
                as_tuple=False,
            )

            for idx_pair in indices:

                token_idx = int(idx_pair[1])

                low = max(
                    0,
                    token_idx - 10,
                )

                high = min(
                    input_ids.size(1) - 1,
                    token_idx + 10,
                )

                nearby = input_ids[
                    0,
                    low : high + 1,
                ]

                replacement = nearby[
                    torch.randint(
                        nearby.numel(),
                        (1,),
                    )
                ]

                input_ids[
                    0,
                    token_idx
                ] = replacement

        # Remaining 10% stay unchanged
        labels[~mask] = -100

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--index",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    idx = args.index
    seed = args.seed

    set_seed(seed)

    conn = get_db_connection()
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

    row = cur.fetchone()

    config_json, train_data, dataset, model_path = row

    output_dir = Path(model_path)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = json.loads(config_json)

    # Load HPs
    num_epochs = config["num_train_epochs"]
    batch_size = config["per_device_train_batch_size"]
    gradient_accumulation_steps = (
        config["gradient_accumulation_steps"]
    )
    learning_rate = config["learning_rate"]
    warmup_ratio = config["warmup_ratio"]
    masking_prob = config["masking_prob"]
    mean_span_length = config["mean_span_length"]

    pipeline = ChronosPipeline.from_pretrained(
        "ddrg/ChronosBERT-Default",
        task="mlm",
    )

    model = pipeline.model.to(DEVICE)
    tokenizer = pipeline.tokenizer

    train_dataset = ImputationDataset(
        train_data,
        tokenizer,
        masking_prob=masking_prob,
        mean_span_length=mean_span_length,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    import math

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    steps_per_epoch = math.ceil(
        len(train_loader)
        / gradient_accumulation_steps
    )

    total_steps = (
        num_epochs * steps_per_epoch
    )

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

        for step, batch in enumerate(train_loader):

            outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
            )

            loss = outputs["loss"]

            # Scale loss because gradients are accumulated
            loss = (
                loss / gradient_accumulation_steps
            )

            loss.backward()

            total_loss += (
                loss.item()
                * gradient_accumulation_steps
            )

            # Optimizer update
            if (
                (step + 1)
                % gradient_accumulation_steps
                == 0
                or (step + 1)
                == len(train_loader)
            ):

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        print(
            f"Epoch {epoch + 1}: "
            f"{total_loss / len(train_loader):.4f}"
        )

    save_path = output_dir

    model.model.save_pretrained(
        save_path
    )

    with open(
        save_path / "training_config.json",
        "w",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    print(
        f"Saved checkpoint: {save_path}"
    )