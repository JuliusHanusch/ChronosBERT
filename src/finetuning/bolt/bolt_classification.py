import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import json
import math
import random
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score


ROOT_DIR = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi"
)

sys.path.insert(0, str(ROOT_DIR.resolve()))
sys.path.insert(0, str((ROOT_DIR / "src").resolve()))
sys.path.insert(0, str((ROOT_DIR / "chronos_pkg/src").resolve()))

from chronos_pkg.src.chronos.chronos_bolt import (
    ChronosBoltModelForForecasting,
)

from chronos_pkg.src.chronos.chronos_bolt_classification import (
    ChronosBoltModelForClassification,
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DATA_ROOT = (
    ROOT_DIR
    / "data/finetuning/UCR_extracted/UCRArchive_2018"
)

EXCLUDED_DATASETS = {
    "Fungi",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    }

SEED = 42

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_config(rng):
    max_gpu_batch = 32
    sampled_batch = int(rng.choice([8, 16, 32, 64, 128]))

    if sampled_batch > max_gpu_batch:
        gradient_accumulation_steps = sampled_batch // max_gpu_batch
        per_device_train_batch_size = max_gpu_batch
    else:
        gradient_accumulation_steps = 1
        per_device_train_batch_size = sampled_batch

    return {
        "num_train_epochs": int(rng.choice([2, 5, 10, 20, 40])),
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": float(
            np.exp(rng.uniform(np.log(5e-6), np.log(1e-3)))
        ),
        "dropout_head": float(rng.uniform(0.0, 0.3)),
        "warmup_ratio": float(rng.uniform(0.0, 0.1)),
        "TrainInnerModel": bool(rng.choice([True, False])),
        "sampled_batch_size": sampled_batch,
    }

def get_datasets():
    datasets = []

    for dataset_dir in sorted(DATA_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue

        name = dataset_dir.name

        if name in EXCLUDED_DATASETS:
            print(f"Skipping {name}")
            continue

        train_path = dataset_dir / f"{name}_TRAIN.tsv"
        test_path = dataset_dir / f"{name}_TEST.tsv"

        if not train_path.exists() or not test_path.exists():
            print(f"[WARNING] Missing files for {name}")
            continue

        datasets.append({
            "name": name,
            "train": str(train_path),
            "test": str(test_path),
        })

    return datasets


def load_ucr_tsv(path):
    df = pd.read_csv(path, sep="\t", header=None)

    labels = df.iloc[:, 0].astype(str).to_numpy()
    X = df.iloc[:, 1:].to_numpy(dtype=np.float32)

    return X, labels


class ClassificationDataset(Dataset):

    def __init__(self, X, y, context_length):
        self.X = X
        self.y = np.asarray(y, dtype=np.int64)
        self.context_length = context_length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        series = torch.tensor(
            self.X[idx],
            dtype=torch.float32,
        )

        # True means that the original value is observed.
        mask = torch.isfinite(series)

        # Replace infinities with NaNs, but preserve missing values
        # until the model has normalized the context.
        series[~mask] = float("nan")

        # Left-pad shorter series to allow batching.
        padding = self.context_length - len(series)

        if padding > 0:
            series = torch.cat([
                torch.full(
                    (padding,),
                    float("nan"),
                    dtype=series.dtype,
                ),
                series,
            ])

            mask = torch.cat([
                torch.zeros(padding, dtype=torch.bool),
                mask,
            ])

        return {
            "context": series,
            "mask": mask,
            "labels": torch.tensor(
                self.y[idx],
                dtype=torch.long,
            ),
        }


def disable_random_patch_augmentation(model):
    config = model.chronos_config

    # Set only attributes that actually exist.
    # Verify these names against the augmentation logic in encode().
    candidate_attributes = [
        "masking_prob",
        "patch_masking_prob",
        "perturbation_prob",
        "patch_perturbation_prob",
    ]

    for attribute in candidate_attributes:
        if hasattr(config, attribute):
            setattr(config, attribute, 0.0)
            print(f"Set {attribute}=0.0")


def build_classification_model(
    model_path,
    num_labels,
    dropout_head,
    freeze_encoder,
):
    """
    Load the pretrained forecasting checkpoint, create a
    classification model with the same configuration, and
    transfer all compatible pretrained weights.

    The classification layer remains randomly initialized.
    """

    pretrained_model = (
        ChronosBoltModelForForecasting.from_pretrained(
            model_path
        )
    )

    disable_random_patch_augmentation(pretrained_model)

    # Reuse the checkpoint's ChronosBolt configuration.
    config = pretrained_model.config

    # The classification class currently uses
    if not hasattr(config, "hidden_size"):
        hidden_size = None

        for attribute in ("hidden_size", "d_model"):
            value = getattr(pretrained_model.config, attribute, None)

            if isinstance(value, int):
                hidden_size = value
                break

        if hidden_size is None:
            position_embeddings = getattr(
                pretrained_model,
                "position_embeddings",
                None,
            )

            if position_embeddings is not None:
                hidden_size = position_embeddings.embedding_dim

        if hidden_size is None:
            patch_embedding = getattr(
                pretrained_model,
                "input_patch_embedding",
                None,
            )

            if patch_embedding is not None:
                hidden_size = getattr(
                    patch_embedding,
                    "out_features",
                    None,
                )

        if hidden_size is None:
            raise ValueError(
                "Could not determine encoder hidden size. "
                "Inspect the model's encoder configuration."
            )

        config.hidden_size = hidden_size

    # Construct the subclass with a new classification head.
    model = ChronosBoltModelForClassification(
        config=config,
        num_labels=num_labels,
        dropout_head=dropout_head,
        pooling="mean",
    )

    # Load pretrained backbone weights into the subclass.
    incompatible = model.load_state_dict(
        pretrained_model.state_dict(),
        strict=False,
    )

    # The classifier weights should be missing from the
    # forecasting checkpoint because they are newly initialized.
    print("Missing checkpoint keys:", incompatible.missing_keys)
    print("Unexpected checkpoint keys:", incompatible.unexpected_keys)

    # Disable augmentation on the final model too.
    disable_random_patch_augmentation(model)

    # Freeze the backbone if requested, but always train
    # the classification head.
    for parameter in model.parameters():
        parameter.requires_grad = not freeze_encoder

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    del pretrained_model

    return model.to(DEVICE)


def train_model(
    model,
    train_loader,
    epochs,
    learning_rate,
    warmup_ratio,
    gradient_accumulation_steps=1,
):
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise ValueError("No trainable model parameters.")

    optimizer = AdamW(
        trainable_parameters,
        lr=learning_rate,
    )

    updates_per_epoch = math.ceil(
        len(train_loader) / gradient_accumulation_steps
    )
    total_steps = epochs * updates_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_samples = 0
        n_batches = len(train_loader)

        for step, batch in enumerate(train_loader):
            context = batch["context"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # Account for a potentially incomplete final accumulation group.
            group_start = (
                step // gradient_accumulation_steps
            ) * gradient_accumulation_steps

            group_size = min(
                gradient_accumulation_steps,
                n_batches - group_start,
            )

            outputs = model(
                context=context,
                mask=mask,
                labels=labels,
            )

            loss = outputs["loss"]

            if loss is None:
                raise RuntimeError(
                    "The classification model returned no loss."
                )

            (loss / group_size).backward()

            should_update = (
                (step + 1) % gradient_accumulation_steps == 0
                or (step + 1) == n_batches
            )

            if should_update:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        average_loss = total_loss / max(total_samples, 1)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- loss: {average_loss:.4f}"
        )

    return model

def run_one_fold(
    X_train,
    y_train,
    X_val,
    y_val,
    config,
    context_length,
    num_labels,
    model_path,
    seed,
):
    set_seed(seed)

    train_dataset = ClassificationDataset(
        X_train, y_train, context_length
    )
    val_dataset = ClassificationDataset(
        X_val, y_val, context_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["per_device_train_batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = build_classification_model(
        model_path=model_path,
        num_labels=num_labels,
        dropout_head=config["dropout_head"],
        freeze_encoder=not config["TrainInnerModel"],
    )

    train_model(
        model=model,
        train_loader=train_loader,
        epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        gradient_accumulation_steps=config[
            "gradient_accumulation_steps"
        ],
    )

    metrics = evaluate_model(model, val_loader)

    del model, train_loader, val_loader

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return metrics


def evaluate_config_cv(
    X,
    y,
    splits,
    config,
    context_length,
    num_labels,
    model_path,
    trial,
):
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        print(f"\nTrial {trial} | Fold {fold}/5")

        metrics = run_one_fold(
            X_train=X[train_idx],
            y_train=y[train_idx],
            X_val=X[val_idx],
            y_val=y[val_idx],
            config=config,
            context_length=context_length,
            num_labels=num_labels,
            model_path=model_path,
            seed=SEED + fold,
        )

        fold_results.append(metrics)

        print(
            f"Validation accuracy: {metrics['accuracy']:.4f} | "
            f"Macro-F1: {metrics['f1']:.4f}"
        )

    return {
        "cv_accuracy_mean": float(
            np.mean([r["accuracy"] for r in fold_results])
        ),
        "cv_accuracy_std": float(
            np.std([r["accuracy"] for r in fold_results])
        ),
        "cv_f1_mean": float(
            np.mean([r["f1"] for r in fold_results])
        ),
        "fold_results": fold_results,
    }


def train_final_model(
    X_train,
    y_train,
    config,
    context_length,
    num_labels,
    model_path,
    seed,
):
    set_seed(seed)

    train_dataset = ClassificationDataset(
        X_train, y_train, context_length
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["per_device_train_batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = build_classification_model(
        model_path=model_path,
        num_labels=num_labels,
        dropout_head=config["dropout_head"],
        freeze_encoder=not config["TrainInnerModel"],
    )

    train_model(
        model=model,
        train_loader=train_loader,
        epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        gradient_accumulation_steps=config[
            "gradient_accumulation_steps"
        ],
    )

    del train_loader

    return model

@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()

    predictions = []
    labels_all = []

    for batch in loader:
        outputs = model(
            context=batch["context"].to(DEVICE),
            mask=batch["mask"].to(DEVICE),
        )

        predictions.extend(
            outputs["logits"].argmax(dim=-1).cpu().numpy()
        )

        labels_all.extend(
            batch["labels"].cpu().numpy()
        )

    return {
        "accuracy": accuracy_score(labels_all, predictions),
        "f1": f1_score(
            labels_all,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }

def main(args):
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {args.model_path}")

    if not Path(args.model_path).exists():
        raise FileNotFoundError(
            f"Checkpoint path does not exist: {args.model_path}"
        )

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"UCR dataset directory does not exist: {DATA_ROOT}"
        )

    # Calculate context length from the checkpoint, not a constant.
    initial_model = ChronosBoltModelForForecasting.from_pretrained(
        args.model_path
    )
    disable_random_patch_augmentation(initial_model)
    context_length = initial_model.chronos_config.context_length

    print(f"Context length: {context_length}")
    del initial_model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    datasets = get_datasets()

    if not 0 <= args.index < len(datasets):
        raise IndexError(
            f"Index {args.index} is out of range. "
            f"Available datasets: {len(datasets)}"
        )

    dataset_info = datasets[args.index]
    dataset_name = dataset_info["name"]

    result_dir = Path(args.output_dir) / dataset_name

    # Skip datasets whose result directory already exists.
    if result_dir.exists():
        print(f"Skipping {dataset_name}: result directory already exists.")
        return

    result_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)


    print("\n" + "=" * 70)
    print(f"Dataset [{args.index}]: {dataset_name}")
    print("=" * 70)

    X_train, y_train_raw = load_ucr_tsv(dataset_info["train"])
    X_test, y_test_raw = load_ucr_tsv(dataset_info["test"])

    # Use the same label mapping for train, validation, and test.
    class_names = sorted(set(y_train_raw.tolist()))
    label_map = {
        label: idx for idx, label in enumerate(class_names)
    }

    unknown_labels = set(y_test_raw.tolist()) - set(label_map)
    if unknown_labels:
        raise ValueError(
            f"Test labels absent from training data: {unknown_labels}"
        )

    y_train = np.asarray(
        [label_map[label] for label in y_train_raw],
        dtype=np.int64,
    )
    y_test = np.asarray(
        [label_map[label] for label in y_test_raw],
        dtype=np.int64,
    )

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    num_labels = len(class_names)

    # Generate identical folds for all sampled configurations.
    splits = list(
        create_cv_splits(y_train, n_splits=5, seed=SEED)
    )

    if any(len(val_idx) == 0 for _, val_idx in splits):
        raise ValueError(
            "At least one validation fold is empty; "
            "five-fold CV cannot be evaluated for this dataset."
        )


    rng = np.random.RandomState(SEED)
    trial_results = []

    for trial in range(1, args.trials + 1):
        config = sample_config(rng)

        print("\n" + "-" * 70)
        print(f"Trial {trial}/{args.trials}")
        print(json.dumps(config, indent=2))
        print("-" * 70)

        cv_metrics = evaluate_config_cv(
            X=X_train,
            y=y_train,
            splits=splits,
            config=config,
            context_length=context_length,
            num_labels=num_labels,
            model_path=args.model_path,
            trial=trial,
        )

        result = {
            "trial": trial,
            **config,
            "cv_accuracy_mean": cv_metrics["cv_accuracy_mean"],
            "cv_accuracy_std": cv_metrics["cv_accuracy_std"],
            "cv_f1_mean": cv_metrics["cv_f1_mean"],
            "fold_results": cv_metrics["fold_results"],
        }
        trial_results.append(result)

        print(
            f"Mean CV accuracy: {result['cv_accuracy_mean']:.4f} "
            f"+/- {result['cv_accuracy_std']:.4f}"
        )

        # Save progress after each trial.
        pd.DataFrame([
            {k: v for k, v in r.items() if k != "fold_results"}
            for r in trial_results
        ]).to_csv(result_dir / "hpo_trials.csv", index=False)

    # Select using validation accuracy only.
    best_result = max(
        trial_results,
        key=lambda result: result["cv_accuracy_mean"],
    )

    config_keys = [
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "dropout_head",
        "warmup_ratio",
        "TrainInnerModel",
        "sampled_batch_size",
    ]
    best_config = {
        key: best_result[key] for key in config_keys
    }

    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print(json.dumps(best_config, indent=2))
    print(f"Mean CV accuracy: {best_result['cv_accuracy_mean']:.4f}")
    print("=" * 70)

    test_dataset = ClassificationDataset(
    X_test, y_test, context_length
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    final_results = []

    for seed in range(42, 47):
        print(f"\nFinal training with seed {seed}")

        set_seed(seed)

        model = train_final_model(
            X_train=X_train,
            y_train=y_train,
            context_length=context_length,
            num_labels=num_labels,
            config=best_config,
            model_path=args.model_path,
            seed=seed,
        )

        test_metrics = evaluate_model(model, test_loader)

        final_results.append({
            "dataset": dataset_name,
            "seed": seed,
            "test_accuracy": test_metrics["accuracy"],
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(final_results)

    avg_row = {
        "dataset": dataset_name,
        "seed": "avg",
        "test_accuracy": results_df["test_accuracy"].mean(),
    }

    results_df = pd.concat(
        [results_df, pd.DataFrame([avg_row])],
        ignore_index=True,
    )

    results_df.to_csv(result_dir / "final_results.csv", index=False)

    print(results_df)
    print(f"Mean test accuracy: {results_df['test_accuracy'].mean():.4f}")
    print(f"Std test accuracy:  {results_df['test_accuracy'].std():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Zero-based index of the dataset returned by get_datasets().",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of random configurations to evaluate.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=(
            "/data/horse/ws/juha972b-AION-BERT-Chronos/"
            "BERTi/BertModel/Bolt/BertBolt200k256/"
            "run-0/checkpoint-final"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(
            ROOT_DIR / "Results/Finetuning/Classification/Bolt_HPO_2"
        ),
    )

    args = parser.parse_args()
    main(args)

