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
from torch.nn.utils.rnn import pad_sequence


root_dir = Path("/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi")

sys.path.append(str(root_dir.resolve()))
sys.path.append(str((root_dir/"src").resolve()))
sys.path.append(str((root_dir/"chronos_pkg/src").resolve()))

from chronos_pkg.src.chronos import ChronosPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DB_PATH = "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi/src/finetuning/similarity/final/similarity_default.db"


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


if __name__=="__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument("--index",type=int,required=True)
    parser.add_argument("--seed",type=int,required=True)
    args=parser.parse_args()

    idx=args.index
    seed=args.seed

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    conn=sqlite3.connect(DB_PATH)
    cur=conn.cursor()
    model_path_column = f"model_path_{seed}"

    cur.execute(
        f"""
        SELECT config,train_data,dataset,task,{model_path_column}
        FROM runs
        WHERE id=?
        """,
        (idx,)
    )

    config_json,train_data,dataset_name,task,output_dir=cur.fetchone()

    output_dir = Path(output_dir)
    config=json.loads(config_json)

    num_epochs = config["num_train_epochs"]
    batch_size = config["per_device_train_batch_size"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    learning_rate = config["learning_rate"]
    dropout_head = config["dropout_head"]
    warmup_ratio = config["warmup_ratio"]
    train_inner = config["TrainInnerModel"]

    pipeline=ChronosPipeline.from_pretrained(
        "ddrg/ChronosBERT-Default",
        task="similarity",
        TrainInnerModel=train_inner,
    )

    model=pipeline.model.to(DEVICE)
    tokenizer=pipeline.tokenizer

    X1, X2, labels = load_pairs(train_data)

    dataset = SimilarityDataset(
        X1,
        X2,
        labels,
        tokenizer
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=similarity_collate,
        generator=generator
    )


    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    import math

    steps_per_epoch = math.ceil(
        len(loader) / gradient_accumulation_steps
    )

    total_steps = (
        steps_per_epoch
        * num_epochs
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

        loss_total = 0.0

        for step, batch in enumerate(loader):

            out = model(
                input_ids_1=batch["input_ids_1"].to(DEVICE),
                attention_mask_1=batch["attention_mask_1"].to(DEVICE),
                input_ids_2=batch["input_ids_2"].to(DEVICE),
                attention_mask_2=batch["attention_mask_2"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
            )

            loss = out["loss"]

            # Keep original loss for logging
            loss_total += loss.item()

            # Scale before accumulating gradients
            loss = loss / gradient_accumulation_steps
            loss.backward()

            # Optimizer update after accumulated gradients
            if (
                (step + 1) % gradient_accumulation_steps == 0
                or (step + 1) == len(loader)
            ):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        print(
            f"Epoch {epoch + 1}: "
            f"{loss_total / len(loader):.4f}"
        )


    save_path = Path(output_dir) 
    save_path.mkdir(parents=True, exist_ok=True)

    model.model.save_pretrained(save_path)

    torch.save(
        model.projection.state_dict(),
        save_path / "projection.pt"
    )

    with open(save_path / "training_config.json","w") as f:
        json.dump(config,f,indent=2)

    print(f"Saved {save_path}")