import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
import torch.nn.functional as F

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)


ROOT_DIR = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi"
)

sys.path.insert(0, str(ROOT_DIR.resolve()))
sys.path.insert(0, str((ROOT_DIR / "src").resolve()))
sys.path.insert(0, str((ROOT_DIR / "chronos_pkg/src").resolve()))

from chronos_pkg.src.chronos.chronos_bolt import (
    ChronosBoltModelForForecasting,
)

# This must be the Siamese Chronos-Bolt similarity model.
from chronos_pkg.src.chronos.chronos_bolt_similarity import (
    ChronosBoltModelForSimilarity,
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DATA_ROOT = (
    ROOT_DIR
    / "data/finetuning/UCR_extracted/UCRArchive_2018"
)

CHECKPOINT = (
    ROOT_DIR
    / "BertModel/Bolt/BertBolt200k256/run-0/checkpoint-final"
)

EXCLUDED_DATASETS = {
    "Fungi"
    "plaid",
    "FordA",
    "HandOutlines",
}

SEED = 42
N_SPLITS = 5
NUM_WORKERS = 0
EVAL_BATCH_SIZE = 32

MAX_GPU_BATCH_SIZE = 32


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_datasets():
    datasets = []

    for dataset_dir in sorted(DATA_ROOT.iterdir()):

        if not dataset_dir.is_dir():
            continue

        name = dataset_dir.name

        if name in EXCLUDED_DATASETS:
            continue

        train_tsv = dataset_dir / f"{name}_TRAIN.tsv"
        test_tsv = dataset_dir / f"{name}_TEST.tsv"

        train_pairs = (
            dataset_dir
            / "similarity"
            / "train_pairs.npz"
        )

        test_pairs = (
            dataset_dir
            / "similarity"
            / "test_pairs.npz"
        )

        if not train_tsv.exists() or not test_tsv.exists():
            continue

        if not train_pairs.exists() or not test_pairs.exists():
            continue

        datasets.append({
            "name": name,
            "dir": dataset_dir,
            "train_pairs": train_pairs,
            "test_pairs": test_pairs,
        })

    return datasets


def load_pairs(path):

    data = np.load(
        path,
        allow_pickle=True,
    )

    pairs_1 = data["pairs_1"]
    pairs_2 = data["pairs_2"]
    labels = np.asarray(data["labels"])

    return pairs_1, pairs_2, labels



def create_cv_splits(y, n_splits=5, seed=42):

    y = np.asarray(y)

    rng = np.random.RandomState(seed)

    fold_val_indices = [
        [] for _ in range(n_splits)
    ]

    for cls in np.unique(y):

        indices = np.where(y == cls)[0]
        rng.shuffle(indices)

        if len(indices) == 1:
            continue

        for i, idx in enumerate(indices):
            fold_val_indices[i % n_splits].append(idx)

    all_indices = np.arange(len(y))

    for fold in range(n_splits):

        val_idx = np.asarray(
            fold_val_indices[fold],
            dtype=int,
        )

        train_idx = np.setdiff1d(
            all_indices,
            val_idx,
        )

        if len(val_idx) == 0:
            raise RuntimeError(
                f"Validation fold {fold + 1} is empty."
            )

        yield train_idx, val_idx


class SimilarityDataset(Dataset):

    def __init__(
        self,
        X1,
        X2,
        y,
        context_length,
    ):

        self.X1 = X1
        self.X2 = X2
        self.y = np.asarray(y)

        self.context_length = context_length

    def __len__(self):
        return len(self.y)

    def prepare_series(self, series):

        series = np.asarray(
            series,
            dtype=np.float32,
        ).reshape(-1)

        # Truncate from the left.
        series = series[-self.context_length:]

        finite_mask = np.isfinite(series)

        series = np.where(
            finite_mask,
            series,
            0.0,
        )

        context = np.zeros(
            self.context_length,
            dtype=np.float32,
        )

        mask = np.zeros(
            self.context_length,
            dtype=np.bool_,
        )

        length = len(series)

        # Right-align the observations.
        context[-length:] = series
        mask[-length:] = finite_mask

        return (
            torch.from_numpy(context),
            torch.from_numpy(mask),
        )

    def __getitem__(self, idx):

        context_1, mask_1 = self.prepare_series(
            self.X1[idx]
        )

        context_2, mask_2 = self.prepare_series(
            self.X2[idx]
        )

        return {
            "context_1": context_1,
            "mask_1": mask_1,
            "context_2": context_2,
            "mask_2": mask_2,
            "labels": torch.tensor(
                self.y[idx],
                dtype=torch.float32,
            ),
        }


def similarity_collate(batch):

    return {
        "context_1": torch.stack(
            [item["context_1"] for item in batch]
        ),
        "mask_1": torch.stack(
            [item["mask_1"] for item in batch]
        ),
        "context_2": torch.stack(
            [item["context_2"] for item in batch]
        ),
        "mask_2": torch.stack(
            [item["mask_2"] for item in batch]
        ),
        "labels": torch.stack(
            [item["labels"] for item in batch]
        ),
    }

def sample_config(rng):

    sampled_batch = int(
        rng.choice([8, 16, 32, 64, 128])
    )

    if sampled_batch > MAX_GPU_BATCH_SIZE:

        gradient_accumulation_steps = (
            sampled_batch // MAX_GPU_BATCH_SIZE
        )

        per_device_train_batch_size = (
            MAX_GPU_BATCH_SIZE
        )

    else:

        gradient_accumulation_steps = 1

        per_device_train_batch_size = sampled_batch

    return {
        "num_train_epochs": int(
            rng.choice([2, 5, 10, 20, 40])
        ),

        "per_device_train_batch_size":
            per_device_train_batch_size,

        "gradient_accumulation_steps":
            gradient_accumulation_steps,

        "sampled_batch_size": sampled_batch,

        "learning_rate": float(
            np.exp(
                rng.uniform(
                    np.log(5e-6),
                    np.log(1e-3),
                )
            )
        ),

        "dropout_head": float(
            rng.uniform(0.0, 0.3)
        ),

        "warmup_ratio": float(
            rng.uniform(0.0, 0.1)
        ),

        "TrainInnerModel": bool(
            rng.choice([True, False])
        ),
    }



def disable_random_augmentation(model):

    for obj in (
        model,
        getattr(model, "config", None),
        getattr(
            getattr(model, "config", None),
            "chronos_config",
            None,
        ),
        getattr(model, "chronos_config", None),
    ):

        if obj is None:
            continue

        for attr in (
            "masking_prob",
            "patch_masking_prob",
            "patch_perturbation_prob",
        ):

            if hasattr(obj, attr):
                setattr(obj, attr, 0.0)


def build_model(config):

    pretrained = ChronosBoltModelForForecasting.from_pretrained(
        str(CHECKPOINT)
    )

    disable_random_augmentation(pretrained)

    model = ChronosBoltModelForSimilarity(
        config=pretrained.config,
        dropout_head=config["dropout_head"],
    )

    # Load pretrained Bolt encoder weights.
    model.load_state_dict(
        pretrained.state_dict(),
        strict=False,
    )

    # Freeze or unfreeze the entire model.
    for parameter in model.parameters():
        parameter.requires_grad = bool(
            config["TrainInnerModel"]
        )

    # Always keep the projection head trainable.
    for parameter in model.projection.parameters():
        parameter.requires_grad = True

    # Also keep the head's dropout module (no parameters).
    disable_random_augmentation(model)

    model.to(DEVICE)

    del pretrained

    return model

def make_loader(
    dataset,
    batch_size,
    shuffle,
    seed,
):

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=similarity_collate,
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    gradient_accumulation_steps,
):

    model.train()

    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    num_batches = len(loader)

    for step, batch in enumerate(loader):

        context_1 = batch["context_1"].to(DEVICE)
        mask_1 = batch["mask_1"].to(DEVICE)

        context_2 = batch["context_2"].to(DEVICE)
        mask_2 = batch["mask_2"].to(DEVICE)

        labels = batch["labels"].to(DEVICE)

        # Expected Siamese model interface:
        # two raw Bolt contexts and their validity masks.
        outputs = model(
            context_1=context_1,
            mask_1=mask_1,
            context_2=context_2,
            mask_2=mask_2,
            labels=labels,
        )

        loss = outputs["loss"]

        if loss is None:
            raise RuntimeError(
                "Similarity model returned no loss."
            )

        total_loss += float(loss.detach().item())

        # Handle incomplete final accumulation group.
        group_start = (
            step // gradient_accumulation_steps
        ) * gradient_accumulation_steps

        group_size = min(
            gradient_accumulation_steps,
            num_batches - group_start,
        )

        (loss / group_size).backward()

        if (
            (step + 1) % gradient_accumulation_steps == 0
            or (step + 1) == num_batches
        ):

            optimizer.step()
            scheduler.step()

            optimizer.zero_grad(set_to_none=True)

    return total_loss / max(num_batches, 1)




def fit_model(
    model,
    train_loader,
    config,
):

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError(
            "No trainable parameters in the similarity model."
        )

    optimizer = AdamW(
        trainable_parameters,
        lr=config["learning_rate"],
    )

    gradient_accumulation_steps = int(
        config["gradient_accumulation_steps"]
    )

    steps_per_epoch = math.ceil(
        len(train_loader)
        / gradient_accumulation_steps
    )

    total_steps = (
        int(config["num_train_epochs"])
        * steps_per_epoch
    )

    warmup_steps = int(
        total_steps * config["warmup_ratio"]
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    for epoch in range(
        int(config["num_train_epochs"])
    ):

        start = time.perf_counter()

        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            gradient_accumulation_steps,
        )

        print(
            f"Epoch {epoch + 1}/"
            f"{config['num_train_epochs']} "
            f"| train_loss={loss:.6f} "
            f"| elapsed={time.perf_counter() - start:.1f}s",
            flush=True,
        )

    return model


@torch.no_grad()
def evaluate_model(model, loader):

    model.eval()

    similarities = []
    labels_all = []

    for batch in loader:

        context_1 = batch["context_1"].to(DEVICE)
        mask_1 = batch["mask_1"].to(DEVICE)

        context_2 = batch["context_2"].to(DEVICE)
        mask_2 = batch["mask_2"].to(DEVICE)

        labels = batch["labels"]

        outputs = model(
            context_1=context_1,
            mask_1=mask_1,
            context_2=context_2,
            mask_2=mask_2,
        )

        z1 = outputs["embeddings_1"]
        z2 = outputs["embeddings_2"]

        # Cosine similarity [-1, 1]
        sim = F.cosine_similarity(
            z1,
            z2,
            dim=-1,
        )

        similarities.append(sim.cpu())
        labels_all.extend(labels.cpu().numpy())

    similarities = torch.cat(similarities).numpy()
    labels_all = np.asarray(labels_all)

    # Convert cosine similarity to [0, 1]
    probs = (similarities + 1) / 2

    preds = (probs > 0.5).astype(int)

    return {
        "accuracy": float(
            accuracy_score(labels_all, preds)
        ),

        "f1": float(
            f1_score(
                labels_all,
                preds,
                zero_division=0,
            )
        ),

        "auroc": float(
            roc_auc_score(labels_all, probs)
        ),
    }


def run_training(
    config,
    train_dataset,
    eval_dataset,
    seed,
):

    set_seed(seed)

    train_loader = make_loader(
        train_dataset,
        config["per_device_train_batch_size"],
        True,
        seed,
    )

    eval_loader = make_loader(
        eval_dataset,
        EVAL_BATCH_SIZE,
        False,
        seed,
    )

    model = build_model(config)

    fit_model(
        model,
        train_loader,
        config,
    )

    metrics = evaluate_model(
        model,
        eval_loader,
    )

    del model
    del train_loader
    del eval_loader

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return metrics



def evaluate_config_cv(
    config,
    X1,
    X2,
    y,
    context_length,
    trial,
):

    fold_results = []

    splits = create_cv_splits(
        y,
        n_splits=N_SPLITS,
        seed=SEED,
    )

    for fold, (train_idx, val_idx) in enumerate(
        splits,
        start=1,
    ):

        print(
            f"\nTrial {trial} | Fold {fold}/{N_SPLITS}",
            flush=True,
        )

        train_dataset = SimilarityDataset(
            X1[train_idx],
            X2[train_idx],
            y[train_idx],
            context_length,
        )

        val_dataset = SimilarityDataset(
            X1[val_idx],
            X2[val_idx],
            y[val_idx],
            context_length,
        )

        metrics = run_training(
            config,
            train_dataset,
            val_dataset,
            seed=SEED + trial * 100 + fold,
        )

        fold_results.append(metrics)

        print(
            f"Fold {fold}: "
            f"Accuracy={metrics['accuracy']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"AUROC={metrics['auroc']:.4f}",
            flush=True,
        )

    return {
        "cv_accuracy_mean": float(
            np.mean([
                result["accuracy"]
                for result in fold_results
            ])
        ),

        "cv_accuracy_std": float(
            np.std([
                result["accuracy"]
                for result in fold_results
            ], ddof=1)
        ),

        "cv_f1_mean": float(
            np.mean([
                result["f1"]
                for result in fold_results
            ])
        ),

        "cv_f1_std": float(
            np.std([
                result["f1"]
                for result in fold_results
            ], ddof=1)
        ),

        "cv_auroc_mean": float(
            np.mean([
                result["auroc"]
                for result in fold_results
            ])
        ),

        "cv_auroc_std": float(
            np.std([
                result["auroc"]
                for result in fold_results
            ], ddof=1)
        ),
    }



def run_dataset(
    dataset_info,
    num_trials,
    output_root,
):

    name = dataset_info["name"]

    result_dir = output_root / name
    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Dataset: {name}")



    X1_train, X2_train, y_train = load_pairs(
        dataset_info["train_pairs"]
    )


    X1_test, X2_test, y_test = load_pairs(
        dataset_info["test_pairs"]
    )


    pretrained = (
        ChronosBoltModelForForecasting.from_pretrained(
            str(CHECKPOINT)
        )
    )

    context_length = (
        pretrained.chronos_config.context_length
    )

    del pretrained

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()



    rng = np.random.RandomState(SEED)

    trial_results = []

    trials_path = (
        result_dir / "hpo_trials.csv"
    )

    for trial in range(1, num_trials + 1):

        config = sample_config(rng)

        print("\n" + "-" * 70)
        print(f"Trial {trial}/{num_trials}")
        print(json.dumps(config, indent=2))
        print("-" * 70)

        cv_metrics = evaluate_config_cv(
            config=config,
            X1=X1_train,
            X2=X2_train,
            y=y_train,
            context_length=context_length,
            trial=trial,
        )

        result = {
            "dataset": name,
            "trial": trial,
            **config,
            **cv_metrics,
        }

        trial_results.append(result)

        pd.DataFrame(
            trial_results
        ).to_csv(
            trials_path,
            index=False,
        )

        print(
            f"\nTrial {trial} results: "
            f"Accuracy={cv_metrics['cv_accuracy_mean']:.4f} | "
            f"F1={cv_metrics['cv_f1_mean']:.4f} | "
            f"AUROC={cv_metrics['cv_auroc_mean']:.4f}",
            flush=True,
        )


    best_result = max(
        trial_results,
        key=lambda result: result["cv_auroc_mean"],
    )

    config_keys = [
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "sampled_batch_size",
        "learning_rate",
        "dropout_head",
        "warmup_ratio",
        "TrainInnerModel",
    ]

    best_config = {
        key: best_result[key]
        for key in config_keys
    }

    with (
        result_dir / "best_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "dataset": name,
                "best_trial": int(
                    best_result["trial"]
                ),
                "cv_auroc_mean": float(
                    best_result["cv_auroc_mean"]
                ),
                "cv_auroc_std": float(
                    best_result["cv_auroc_std"]
                ),
                "config": best_config,
            },
            f,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print(json.dumps(best_config, indent=2))
    print(
        f"Mean CV AUROC: "
        f"{best_result['cv_auroc_mean']:.4f}"
    )
    print("=" * 70)


    train_dataset = SimilarityDataset(
        X1_train,
        X2_train,
        y_train,
        context_length,
    )

    test_dataset = SimilarityDataset(
        X1_test,
        X2_test,
        y_test,
        context_length,
    )

    final_results = []

    for seed in range(42, 47):

        print(
            f"\nFinal training with seed {seed}",
            flush=True,
        )

        metrics = run_training(
            config=best_config,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            seed=seed,
        )

        final_results.append({
            "dataset": name,
            "seed": seed,
            "test_accuracy": metrics["accuracy"],
            "test_f1": metrics["f1"],
            "test_auroc": metrics["auroc"],
        })

        print(
            f"Test accuracy={metrics['accuracy']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"AUROC={metrics['auroc']:.4f}",
            flush=True,
        )


    results_df = pd.DataFrame(
        final_results
    )

    avg_row = {
        "dataset": name,
        "seed": "avg",
        "test_accuracy": results_df[
            "test_accuracy"
        ].mean(),
        "test_f1": results_df[
            "test_f1"
        ].mean(),
        "test_auroc": results_df[
            "test_auroc"
        ].mean(),
    }

    std_row = {
        "dataset": name,
        "seed": "std",
        "test_accuracy": results_df[
            "test_accuracy"
        ].std(ddof=1),
        "test_f1": results_df[
            "test_f1"
        ].std(ddof=1),
        "test_auroc": results_df[
            "test_auroc"
        ].std(ddof=1),
    }

    results_df = pd.concat(
        [
            results_df,
            pd.DataFrame([avg_row, std_row]),
        ],
        ignore_index=True,
    )

    results_df.to_csv(
        result_dir / "final_results.csv",
        index=False,
    )

    print("\nFinal results:")
    print(
        results_df.to_string(index=False)
    )

    print(
        f"\nMean test AUROC: "
        f"{avg_row['test_auroc']:.4f}"
    )

    print(
        f"Std test AUROC: "
        f"{std_row['test_auroc']:.4f}"
    )



def main(args):

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Chronos-Bolt checkpoint not found: {CHECKPOINT}"
        )

    datasets = get_datasets()

    if not datasets:
        raise RuntimeError(
            f"No datasets with train_pairs.npz and "
            f"test_pairs.npz found under {DATA_ROOT}"
        )

    print(f"Device: {DEVICE}")
    print(f"Found {len(datasets)} datasets:")

    for idx, info in enumerate(datasets):

        print(
            f"  [{idx}] {info['name']} "
            f"— {info['dir']}"
        )

    if not 0 <= args.index < len(datasets):

        raise ValueError(
            f"Dataset index {args.index} is invalid. "
            f"Valid range: 0 to {len(datasets) - 1}."
        )

    dataset_info = datasets[args.index]

    output_root = Path(args.output_dir)
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_dataset(
        dataset_info=dataset_info,
        num_trials=args.trials,
        output_root=output_root,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Zero-based index of the dataset.",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of random HPO configurations.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(
            ROOT_DIR
            / "Results/Finetuning/Similarity/Bolt_HPO"
        ),
    )

    args = parser.parse_args()

    main(args)