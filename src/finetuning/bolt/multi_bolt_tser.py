import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gluonts.dataset.arrow import ArrowFile
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


# ============================================================
# Paths and settings
# ============================================================

ROOT_DIR = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi"
)

DATA_ROOT = ROOT_DIR / "data/finetuning/TSER"

OUTPUT_DIR = (
    ROOT_DIR
    / "Results/Finetuning/TSER/Bolt_Multivariate_HPO"
)

CHECKPOINT = (
    ROOT_DIR
    / "BertModel/Bolt/BertBolt200k256/run-0/checkpoint-final"
)

NUM_TRIALS = 100
MAX_GPU_BATCH = 32
SEED = 42
FINAL_SEEDS = range(42, 47)
EVAL_BATCH_SIZE = 32
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "chronos_pkg/src"))

from chronos.chronos_bolt import ChronosBoltModelForForecasting
from chronos.chronos_bolt_tser import (
    ChronosBoltModelForTSERMultivariate,
)



def load_existing_trial_results(result_dir):
    """
    Load completed trials from the central CSV and individual JSON files.

    Individual JSON files take precedence over CSV entries.
    """
    results = {}

    trials_path = result_dir / "hpo_trials.csv"

    if trials_path.exists():
        df = pd.read_csv(trials_path)

        if "trial" in df.columns:
            for row in df.to_dict("records"):
                if pd.notna(row.get("trial")):
                    trial = int(row["trial"])
                    results[trial] = row

    trials_dir = result_dir / "trials"

    if trials_dir.exists():
        for path in trials_dir.glob("trial_*.json"):
            with path.open("r", encoding="utf-8") as f:
                row = json.load(f)

            trial = int(row["trial"])
            results[trial] = row

    return results


def save_trial_result(result_dir, result):
    """
    Save one trial to its own JSON file.

    A temporary file followed by a rename prevents incomplete
    JSON files from being treated as completed trials.
    """
    trials_dir = result_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    trial = int(result["trial"])

    path = trials_dir / f"trial_{trial:03d}.json"
    temp_path = trials_dir / f"trial_{trial:03d}.tmp"

    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    temp_path.replace(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_config(rng, max_gpu_batch=MAX_GPU_BATCH):
    sampled_batch = int(
        rng.choice([8, 16, 32, 64, 128])
    )

    if sampled_batch > max_gpu_batch:
        batch_size = max_gpu_batch
        grad_accum = sampled_batch // max_gpu_batch
    else:
        batch_size = sampled_batch
        grad_accum = 1

    return {
        "num_train_epochs": int(
            rng.choice([2, 5, 10, 20, 40])
        ),
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
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
        "loss_type": str(
            rng.choice(["mae", "mse"])
        ),
    }


# ============================================================
# Find datasets
# ============================================================

def find_datasets(data_root):
    required_files = [
        "train_small_multi.arrow",
        "eval_multi.arrow",
        "train_multi.arrow",
        "test_multi.arrow",
    ]

    if not data_root.is_dir():
        raise FileNotFoundError(
            f"TSER data directory not found: {data_root}"
        )

    datasets = []

    for small_train in sorted(
        data_root.rglob("train_small_multi.arrow")
    ):
        dataset_dir = small_train.parent

        paths = {
            filename: dataset_dir / filename
            for filename in required_files
        }

        missing = [
            filename
            for filename, path in paths.items()
            if not path.is_file()
        ]

        if missing:
            print(
                f"Skipping {dataset_dir}: "
                f"missing {', '.join(missing)}"
            )
            continue

        datasets.append({
            "name": dataset_dir.name,
            "dir": dataset_dir,
            **paths,
        })

    return datasets


# ============================================================
# Load multivariate Arrow data
# ============================================================

def load_arrow(path):
    series_list = []
    labels = []

    for entry in ArrowFile(path):
        target = np.asarray(
            entry["target"],
            dtype=np.float32,
        )

        if target.ndim != 2:
            raise ValueError(
                f"Expected [channels, time], got {target.shape}"
            )

        # Keep the original [channels, time] orientation.
        series_list.append(target)

        if "label" in entry:
            label = entry["label"]
        elif "y" in entry:
            label = entry["y"]
        else:
            raise KeyError(
                f"No regression label found in {path}"
            )

        label = float(label)

        if not np.isfinite(label):
            raise ValueError(
                f"Non-finite regression label in {path}"
            )

        labels.append(label)

    if not series_list:
        raise ValueError(
            f"No samples found in {path}"
        )

    return series_list, np.asarray(
        labels,
        dtype=np.float32,
    )


# ============================================================
# Multivariate dataset
# ============================================================

class TSERMultivariateDataset(Dataset):
    def __init__(
        self,
        series_list,
        labels,
        context_length,
        expected_channels=None,
    ):
        self.series_list = series_list
        self.labels = np.asarray(
            labels,
            dtype=np.float32,
        )
        self.context_length = context_length

        if len(series_list) != len(self.labels):
            raise ValueError(
                "Number of series and labels does not match."
            )

        self.n_channels = (
            int(expected_channels)
            if expected_channels is not None
            else int(series_list[0].shape[0])
        )

        for i, series in enumerate(series_list):
            if series.ndim != 2:
                raise ValueError(
                    f"Sample {i} must have shape "
                    "[channels, time]."
                )

            if series.shape[0] != self.n_channels:
                raise ValueError(
                    f"Sample {i} has {series.shape[0]} channels; "
                    f"expected {self.n_channels}."
                )

    def __len__(self):
        return len(self.series_list)

    def __getitem__(self, idx):
        series = np.asarray(
            self.series_list[idx],
            dtype=np.float32,
        )

        # Truncate from the left if the sequence is too long.
        series = series[:, -self.context_length:]

        finite_mask = np.isfinite(series)
        series = np.where(finite_mask, series, 0.0)

        context = np.zeros(
            (self.n_channels, self.context_length),
            dtype=np.float32,
        )

        mask = np.zeros(
            (self.n_channels, self.context_length),
            dtype=np.bool_,
        )

        length = series.shape[1]

        context[:, -length:] = series
        mask[:, -length:] = finite_mask

        return {
            "context": torch.from_numpy(context),
            "mask": torch.from_numpy(mask),
            "labels": torch.tensor(
                self.labels[idx],
                dtype=torch.float32,
            ),
        }


# ============================================================
# Model setup
# ============================================================

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


def build_model(config, n_channels):
    pretrained = (
        ChronosBoltModelForForecasting.from_pretrained(
            str(CHECKPOINT)
        )
    )

    disable_random_augmentation(pretrained)

    model = ChronosBoltModelForTSERMultivariate(
        config=pretrained.config,
        n_channels=n_channels,
        dropout_head=config["dropout_head"],
        loss_type=config["loss_type"],
    )

    # Load the pretrained Bolt backbone.
    # The multivariate fusion head is initialized separately.
    missing, unexpected = model.load_state_dict(
        pretrained.state_dict(),
        strict=False,
    )

    missing_backbone = [
        key for key in missing
        if not key.startswith("fusion.")
    ]

    unexpected_backbone = [
        key for key in unexpected
        if not key.startswith(("fusion.", "regressor."))
    ]

    if missing_backbone or unexpected_backbone:
        raise RuntimeError(
            "Pretrained checkpoint mismatch.\n"
            f"Missing backbone keys: {missing_backbone}\n"
            f"Unexpected checkpoint keys: {unexpected_backbone}"
        )

    # Train or freeze the pretrained model according to HPO.
    for parameter in model.parameters():
        parameter.requires_grad = bool(
            config["TrainInnerModel"]
        )

    # Always train the fusion head.
    for parameter in model.fusion.parameters():
        parameter.requires_grad = True

    disable_random_augmentation(model)
    model.to(DEVICE)

    del pretrained

    return model


# ============================================================
# Data loaders
# ============================================================

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
    )


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    grad_accum_steps,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    num_batches = len(loader)

    for step, batch in enumerate(loader):
        context = batch["context"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            context=context,
            mask=mask,
            labels=labels,
        )

        loss = outputs["loss"]

        if loss is None:
            raise RuntimeError(
                "Multivariate TSER model returned no loss."
            )

        total_loss += float(loss.detach().item())

        group_start = (
            step // grad_accum_steps
        ) * grad_accum_steps

        group_size = min(
            grad_accum_steps,
            num_batches - group_start,
        )

        (loss / group_size).backward()

        if (
            (step + 1) % grad_accum_steps == 0
            or (step + 1) == num_batches
        ):
            torch.nn.utils.clip_grad_norm_(
                [
                    p for p in model.parameters()
                    if p.requires_grad
                ],
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    return total_loss / max(num_batches, 1)


def fit_model(model, train_loader, config):
    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    if not trainable_params:
        raise RuntimeError(
            "No trainable model parameters."
        )

    optimizer = AdamW(
        trainable_params,
        lr=config["learning_rate"],
    )

    grad_accum = config["gradient_accumulation_steps"]

    steps_per_epoch = math.ceil(
        len(train_loader) / grad_accum
    )

    total_steps = (
        config["num_train_epochs"] * steps_per_epoch
    )

    warmup_steps = int(
        total_steps * config["warmup_ratio"]
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    for epoch in range(config["num_train_epochs"]):
        start = time.perf_counter()

        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            grad_accum,
        )

        print(
            f"Epoch {epoch + 1}/"
            f"{config['num_train_epochs']} "
            f"| train_loss={loss:.6f} "
            f"| elapsed={time.perf_counter() - start:.1f}s",
            flush=True,
        )

    return model


# ============================================================
# Evaluation
# ============================================================

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
            outputs["logits"]
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .tolist()
        )

        labels_all.extend(
            batch["labels"]
            .cpu()
            .numpy()
            .reshape(-1)
            .tolist()
        )

    predictions = np.asarray(
        predictions,
        dtype=np.float32,
    )
    labels_all = np.asarray(
        labels_all,
        dtype=np.float32,
    )

    return {
        "mae": float(
            mean_absolute_error(
                labels_all,
                predictions,
            )
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    labels_all,
                    predictions,
                )
            )
        ),
    }


def run_training(
    config,
    train_dataset,
    eval_dataset,
    n_channels,
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

    model = build_model(config, n_channels)

    fit_model(model, train_loader, config)
    metrics = evaluate_model(model, eval_loader)

    del model, train_loader, eval_loader

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return metrics



def run_dataset(dataset_info, config_index=None):
    name = dataset_info["name"]

    result_dir = OUTPUT_DIR / name
    result_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Dataset: {name}")
    print(f"Directory: {dataset_info['dir']}")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    print("Loading multivariate Arrow datasets...")

    x_train_small, y_train_small = load_arrow(
        dataset_info["train_small_multi.arrow"]
    )
    x_val, y_val = load_arrow(
        dataset_info["eval_multi.arrow"]
    )
    x_train_full, y_train_full = load_arrow(
        dataset_info["train_multi.arrow"]
    )
    x_test, y_test = load_arrow(
        dataset_info["test_multi.arrow"]
    )

    n_channels = x_train_small[0].shape[0]

    # Ensure all splits use the same number of channels.
    for split_name, split in (
        ("validation", x_val),
        ("full train", x_train_full),
        ("test", x_test),
    ):
        for i, series in enumerate(split):
            if series.shape[0] != n_channels:
                raise ValueError(
                    f"{name}: sample {i} in {split_name} has "
                    f"{series.shape[0]} channels; "
                    f"expected {n_channels}."
                )

    pretrained = (
        ChronosBoltModelForForecasting.from_pretrained(
            str(CHECKPOINT)
        )
    )

    context_length = pretrained.chronos_config.context_length
    del pretrained

    print(f"Number of channels: {n_channels}")
    print(
        f"Samples: train_small={len(x_train_small)}, "
        f"val={len(x_val)}, "
        f"train_full={len(x_train_full)}, "
        f"test={len(x_test)}"
    )
    print(f"Model context length: {context_length}")

    train_small_ds = TSERMultivariateDataset(
        x_train_small,
        y_train_small,
        context_length,
        expected_channels=n_channels,
    )

    val_ds = TSERMultivariateDataset(
        x_val,
        y_val,
        context_length,
        expected_channels=n_channels,
    )

    train_full_ds = TSERMultivariateDataset(
        x_train_full,
        y_train_full,
        context_length,
        expected_channels=n_channels,
    )

    test_ds = TSERMultivariateDataset(
        x_test,
        y_test,
        context_length,
        expected_channels=n_channels,
    )

    # --------------------------------------------------------
    # HPO: train on train_small_multi, validate on eval_multi
    # --------------------------------------------------------

    trial_results = load_existing_trial_results(result_dir)

    print(
        f"Found {len(trial_results)} completed trials.",
        flush=True,
    )

    # --------------------------------------------------------
    # Individual trial mode (SLURM array)
    # --------------------------------------------------------

    if config_index is not None:

        trial = int(config_index)

        if not 1 <= trial <= NUM_TRIALS:
            raise ValueError(
                f"Trial index must be between 1 and {NUM_TRIALS}."
            )

        if trial in trial_results:
            print(
                f"Trial {trial} already exists. Skipping.",
                flush=True,
            )
            return

        # Reconstruct the deterministic configuration for this trial.
        trial_rng = np.random.RandomState(SEED)

        config = None

        for _ in range(trial):
            config = sample_config(trial_rng)

        print(
            f"\nRunning HPO trial {trial}/{NUM_TRIALS}",
            flush=True,
        )
        print(json.dumps(config, indent=2))

        val_metrics = run_training(
            config=config,
            train_dataset=train_small_ds,
            eval_dataset=val_ds,
            n_channels=n_channels,
            seed=SEED + trial,
        )

        result = {
            "dataset": name,
            "trial": trial,
            **config,
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
        }

        save_trial_result(result_dir, result)

        print(
            f"Trial {trial} saved.\n"
            f"Validation MAE: {val_metrics['mae']:.6f} | "
            f"Validation RMSE: {val_metrics['rmse']:.6f}",
            flush=True,
        )

        # Individual trial jobs must not perform final training.
        return

    # --------------------------------------------------------
    # Full HPO mode: run all missing trials
    # --------------------------------------------------------

    rng = np.random.RandomState(SEED)

    for trial in range(1, NUM_TRIALS + 1):

        # Always sample to preserve the original deterministic
        # configuration sequence, including completed trials.
        config = sample_config(rng)

        if trial in trial_results:
            print(
                f"Trial {trial}/{NUM_TRIALS} already completed. "
                "Skipping.",
                flush=True,
            )
            continue

        print(
            f"\nHPO trial {trial}/{NUM_TRIALS}",
            flush=True,
        )
        print(json.dumps(config, indent=2))

        val_metrics = run_training(
            config=config,
            train_dataset=train_small_ds,
            eval_dataset=val_ds,
            n_channels=n_channels,
            seed=SEED + trial,
        )

        result = {
            "dataset": name,
            "trial": trial,
            **config,
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
        }

        save_trial_result(result_dir, result)
        trial_results[trial] = result

        print(
            f"Validation MAE: {val_metrics['mae']:.6f} | "
            f"Validation RMSE: {val_metrics['rmse']:.6f}",
            flush=True,
        )

    # --------------------------------------------------------
    # Check that all trials are available
    # --------------------------------------------------------

    missing_trials = [
        trial
        for trial in range(1, NUM_TRIALS + 1)
        if trial not in trial_results
    ]

    if missing_trials:
        raise RuntimeError(
            f"Missing HPO trials for {name}: {missing_trials}"
        )

    # --------------------------------------------------------
    # Aggregate all trial results into the central CSV
    # --------------------------------------------------------

    trial_rows = [
        trial_results[trial]
        for trial in range(1, NUM_TRIALS + 1)
    ]

    trial_df = pd.DataFrame(trial_rows).sort_values("trial")

    trials_path = result_dir / "hpo_trials.csv"

    trial_df.to_csv(
        trials_path,
        index=False,
    )

    print(
        f"\nCollected {len(trial_df)} HPO trials into "
        f"{trials_path}",
        flush=True,
    )

    # --------------------------------------------------------
    # Select the best configuration by validation RMSE
    # --------------------------------------------------------

    best_result = min(
        trial_rows,
        key=lambda result: float(result["val_rmse"]),
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
        "loss_type",
    ]

    best_config = {
        key: best_result[key]
        for key in config_keys
    }

    # Convert values to the expected Python types.
    for key in (
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "sampled_batch_size",
    ):
        best_config[key] = int(best_config[key])

    for key in (
        "learning_rate",
        "dropout_head",
        "warmup_ratio",
    ):
        best_config[key] = float(best_config[key])

    if isinstance(best_config["TrainInnerModel"], str):
        best_config["TrainInnerModel"] = (
            best_config["TrainInnerModel"].strip().lower()
            in ("true", "1")
        )
    else:
        best_config["TrainInnerModel"] = bool(
            best_config["TrainInnerModel"]
        )

    best_config["loss_type"] = str(
        best_config["loss_type"]
    ).lower()

    best_config_path = result_dir / "best_config.json"

    with best_config_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "dataset": name,
                "best_trial": int(best_result["trial"]),
                "val_mae": float(best_result["val_mae"]),
                "val_rmse": float(best_result["val_rmse"]),
                "n_channels": n_channels,
                "context_length": context_length,
                "config": best_config,
            },
            f,
            indent=2,
        )

    print("\nBest multivariate configuration:")
    print(json.dumps(best_config, indent=2))
    print(
        f"Best validation RMSE: "
        f"{float(best_result['val_rmse']):.6f}"
    )

    # --------------------------------------------------------
    # Final evaluation: train on train_multi, test on test_multi
    # --------------------------------------------------------

    final_rows = []

    for seed in FINAL_SEEDS:
        print(f"\nFinal full-training run with seed {seed}")

        metrics = run_training(
            config=best_config,
            train_dataset=train_full_ds,
            eval_dataset=test_ds,
            n_channels=n_channels,
            seed=seed,
        )

        row = {
            "dataset": name,
            "seed": seed,
            "n_channels": n_channels,
            "test_mae": metrics["mae"],
            "test_rmse": metrics["rmse"],
            "best_trial": int(best_result["trial"]),
            "val_mae": float(best_result["val_mae"]),
            "val_rmse": float(best_result["val_rmse"]),
        }

        final_rows.append(row)

        pd.DataFrame(final_rows).to_csv(
            result_dir / "final_results_seeds.csv",
            index=False,
        )

        print(
            f"Test MAE: {metrics['mae']:.6f} | "
            f"Test RMSE: {metrics['rmse']:.6f}",
            flush=True,
        )

    seed_df = pd.DataFrame(final_rows)

    avg_row = {
        "dataset": name,
        "seed": "avg",
        "n_channels": n_channels,
        "test_mae": seed_df["test_mae"].mean(),
        "test_rmse": seed_df["test_rmse"].mean(),
        "test_mae_std": seed_df["test_mae"].std(ddof=1),
        "test_rmse_std": seed_df["test_rmse"].std(ddof=1),
        "best_trial": int(best_result["trial"]),
        "val_mae": float(best_result["val_mae"]),
        "val_rmse": float(best_result["val_rmse"]),
    }

    final_df = pd.concat(
        [seed_df, pd.DataFrame([avg_row])],
        ignore_index=True,
    )

    final_df.to_csv(
        result_dir / "final_results.csv",
        index=False,
    )

    print("\nFinal results:")
    print(final_df.to_string(index=False))



def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {CHECKPOINT}"
        )

    datasets = find_datasets(DATA_ROOT)

    if not datasets:
        raise RuntimeError(
            "No datasets containing the required multivariate Arrow "
            f"files were found under {DATA_ROOT}."
        )

    print(f"Device: {DEVICE}")
    print(f"Found {len(datasets)} datasets:")

    for idx, info in enumerate(datasets):
        print(f"  [{idx}] {info['name']} — {info['dir']}")

    if len(sys.argv) > 3:
        raise SystemExit(
            "Usage: python tser_bolt_multivariate_hpo.py "
            "[dataset_index] [trial_index]"
        )

    dataset_index = None
    config_index = None

    if len(sys.argv) >= 2:
        try:
            dataset_index = int(sys.argv[1])
        except ValueError:
            raise SystemExit(
                "Dataset index must be an integer."
            )

        if not 0 <= dataset_index < len(datasets):
            raise SystemExit(
                f"Dataset index {dataset_index} is out of range. "
                f"Valid indices: 0 to {len(datasets) - 1}."
            )

        datasets = [datasets[dataset_index]]

    if len(sys.argv) == 3:
        try:
            config_index = int(sys.argv[2])
        except ValueError:
            raise SystemExit(
                "Trial index must be an integer."
            )

        if not 1 <= config_index <= NUM_TRIALS:
            raise SystemExit(
                f"Trial index must be between 1 and {NUM_TRIALS}."
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for dataset_info in datasets:
        run_dataset(
            dataset_info,
            config_index=config_index,
        )


if __name__ == "__main__":
    main()
