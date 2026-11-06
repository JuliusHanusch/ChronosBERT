
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



ROOT_DIR = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi"
)

DATA_ROOT = ROOT_DIR / "data/finetuning/TSER"

OUTPUT_DIR = ROOT_DIR / "Results/Finetuning/TSER/Bolt_HPO_2"

CHECKPOINT = (
    ROOT_DIR
    / "BertModel/Bolt/BertBolt200k256/run-0/checkpoint-final"
)

NUM_TRIALS = 100
MAX_GPU_BATCH = 32
SEED = 42
FINAL_SEEDS = range(42, 47)
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)



sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "chronos_pkg/src"))

from chronos.chronos_bolt import ChronosBoltModelForForecasting
from chronos.chronos_bolt_tser import ChronosBoltModelForTSER

def load_existing_trial_results(result_dir):
    """
    Load completed trials from the central CSV and individual JSON files.

    Individual JSON files take precedence over CSV entries.
    """
    results = {}

    trials_path = result_dir / "hpo_trials.csv"

    # Load central CSV, if available.
    if trials_path.exists():
        df = pd.read_csv(trials_path)

        if "trial" in df.columns:
            for row in df.to_dict("records"):
                trial = int(row["trial"])
                results[trial] = row

    # Load individual trial files.
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

    The temporary file and rename prevent incomplete JSON files
    from being mistaken for completed trials.
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
    sampled_batch = int(rng.choice([8, 16, 32, 64, 128]))

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
                    np.log(1e-3)
                )
            )
        ),
        "dropout_head": float(rng.uniform(0.0, 0.3)),
        "warmup_ratio": float(rng.uniform(0.0, 0.1)),
        "TrainInnerModel": bool(rng.choice([True, False])),
        "loss_type": str(rng.choice(["mae", "mse"])),
    }


def find_datasets(data_root):
    required_files = [
        "train_small_uni.arrow",
        "eval_uni.arrow",
        "train_uni.arrow",
        "test_uni.arrow",
    ]

    datasets = []

    if not data_root.is_dir():
        raise FileNotFoundError(
            f"TSER data directory not found: {data_root}"
        )

    for small_train in sorted(
        data_root.rglob("train_small_uni.arrow")
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
# Load Arrow data
# ============================================================

def load_arrow(path):
    series_list = []
    labels = []

    for entry in ArrowFile(path):
        target = np.asarray(
            entry["target"],
            dtype=np.float32
        ).reshape(-1)

        if "label" in entry:
            label = entry["label"]
        elif "y" in entry:
            label = entry["y"]
        else:
            raise KeyError(
                f"No regression label found in {path}. "
                "Expected 'label' or 'y'."
            )

        label = float(label)

        if not np.isfinite(label):
            raise ValueError(
                f"Non-finite regression label in {path}"
            )

        if target.size == 0:
            raise ValueError(
                f"Empty time series encountered in {path}"
            )

        series_list.append(target)
        labels.append(label)

    if not series_list:
        raise ValueError(f"No samples found in {path}")

    return series_list, np.asarray(
        labels,
        dtype=np.float32
    )


class TSERDataset(Dataset):
    def __init__(
        self,
        series_list,
        labels,
        context_length
    ):
        self.series_list = series_list
        self.labels = np.asarray(labels, dtype=np.float32)
        self.context_length = context_length

    def __len__(self):
        return len(self.series_list)

    def __getitem__(self, idx):
        series = np.asarray(
            self.series_list[idx],
            dtype=np.float32
        ).reshape(-1)

        series = series[-self.context_length:]

        finite_mask = np.isfinite(series)
        series = np.where(finite_mask, series, 0.0)

        context = np.zeros(
            self.context_length,
            dtype=np.float32
        )
        mask = np.zeros(
            self.context_length,
            dtype=np.bool_
        )

        length = len(series)

        context[-length:] = series
        mask[-length:] = finite_mask

        return {
            "context": torch.from_numpy(context),
            "mask": torch.from_numpy(mask),
            "labels": torch.tensor(
                self.labels[idx],
                dtype=torch.float32
            ),
        }


def disable_random_augmentation(model):
    for obj in (
        model,
        getattr(model, "config", None),
        getattr(
            getattr(model, "config", None),
            "chronos_config",
            None
        ),
        getattr(model, "chronos_config", None),
    ):
        if obj is None:
            continue

        for attr in (
            "masking_prob",
            "patch_masking_prob",
            "patch_perturbation_prob"
        ):
            if hasattr(obj, attr):
                setattr(obj, attr, 0.0)


def build_model(config):
    pretrained = (
        ChronosBoltModelForForecasting.from_pretrained(
            str(CHECKPOINT)
        )
    )

    disable_random_augmentation(pretrained)


    model = ChronosBoltModelForTSER(
        config=pretrained.config,
        dropout_head=float(config["dropout_head"]),
        loss_type=config["loss_type"],
    )

    missing, unexpected = model.load_state_dict(
        pretrained.state_dict(),
        strict=False
    )

    unexpected_backbone = [
        key for key in unexpected
        if not key.startswith(("regressor.", "classifier."))
    ]

    missing_backbone = [
        key for key in missing
        if not key.startswith(("regressor.", "classifier."))
    ]

    if unexpected_backbone or missing_backbone:
        raise RuntimeError(
            "Pretrained checkpoint mismatch.\n"
            f"Missing backbone keys: {missing_backbone}\n"
            f"Unexpected checkpoint keys: {unexpected_backbone}"
        )

    for parameter in model.parameters():
        parameter.requires_grad = bool(
            config["TrainInnerModel"]
        )

    # The regression head is always trained.
    for parameter in model.regressor.parameters():
        parameter.requires_grad = True

    disable_random_augmentation(model)
    model.to(DEVICE)

    del pretrained

    return model


# ============================================================
# Training and evaluation
# ============================================================

def make_loader(dataset, batch_size, shuffle, seed):
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


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    grad_accum_steps
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
            labels=labels
        )

        loss = outputs["loss"]

        if loss is None:
            raise RuntimeError(
                "TSER model returned no training loss."
            )

        total_loss += float(loss.detach().item())

        group_start = (
            step // grad_accum_steps
        ) * grad_accum_steps

        group_size = min(
            grad_accum_steps,
            num_batches - group_start
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
                max_norm=1.0
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
        raise RuntimeError("No trainable model parameters.")

    optimizer = AdamW(
        trainable_params,
        lr=float(config["learning_rate"])
    )

    grad_accum = int(
        config["gradient_accumulation_steps"]
    )

    steps_per_epoch = math.ceil(
        len(train_loader) / grad_accum
    )

    total_steps = (
        int(config["num_train_epochs"])
        * steps_per_epoch
    )

    warmup_steps = int(
        total_steps * float(config["warmup_ratio"])
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
            grad_accum
        )

        print(
            f"Epoch {epoch + 1}/"
            f"{config['num_train_epochs']} "
            f"| train_loss={loss:.6f} "
            f"| elapsed={time.perf_counter() - start:.1f}s",
            flush=True
        )

    return model


@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()

    predictions = []
    labels_all = []

    for batch in loader:
        outputs = model(
            context=batch["context"].to(DEVICE),
            mask=batch["mask"].to(DEVICE)
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
            .numpy()
            .reshape(-1)
            .tolist()
        )

    predictions = np.asarray(
        predictions,
        dtype=np.float32
    )

    labels_all = np.asarray(
        labels_all,
        dtype=np.float32
    )

    return {
        "mae": float(
            mean_absolute_error(labels_all, predictions)
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(labels_all, predictions)
            )
        ),
    }


def run_training(
    config,
    train_dataset,
    eval_dataset,
    seed
):
    set_seed(seed)

    train_loader = make_loader(
        train_dataset,
        config["per_device_train_batch_size"],
        True,
        seed
    )

    eval_loader = make_loader(
        eval_dataset,
        32,
        False,
        seed
    )

    model = build_model(config)

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

    print("Loading Arrow datasets...")

    x_train_small, y_train_small = load_arrow(
        dataset_info["train_small_uni.arrow"]
    )

    x_val, y_val = load_arrow(
        dataset_info["eval_uni.arrow"]
    )

    x_train_full, y_train_full = load_arrow(
        dataset_info["train_uni.arrow"]
    )

    x_test, y_test = load_arrow(
        dataset_info["test_uni.arrow"]
    )

    # Get the context length from the pretrained Chronos-Bolt model.
    pretrained = ChronosBoltModelForForecasting.from_pretrained(
        str(CHECKPOINT)
    )

    context_length = pretrained.chronos_config.context_length

    del pretrained

    print(f"Model context length: {context_length}")

    train_small_ds = TSERDataset(
        x_train_small,
        y_train_small,
        context_length
    )

    val_ds = TSERDataset(
        x_val,
        y_val,
        context_length
    )

    train_full_ds = TSERDataset(
        x_train_full,
        y_train_full,
        context_length
    )

    test_ds = TSERDataset(
        x_test,
        y_test,
        context_length
    )

    print(
        f"Samples: train_small={len(train_small_ds)}, "
        f"val={len(val_ds)}, "
        f"train_full={len(train_full_ds)}, "
        f"test={len(test_ds)}"
    )

    # --------------------------------------------------------
    # Load existing trial results
    # --------------------------------------------------------

    trial_results = load_existing_trial_results(result_dir)

    completed_trials = set(trial_results.keys())

    print(
        f"Found {len(completed_trials)} completed trials: "
        f"{sorted(completed_trials)}"
    )

    # --------------------------------------------------------
    # Single configuration mode
    # --------------------------------------------------------

    if config_index is not None:

        trial = int(config_index)

        if not 1 <= trial <= NUM_TRIALS:
            raise ValueError(
                f"Trial index must be between 1 and {NUM_TRIALS}, "
                f"got {trial}."
            )

        if trial in completed_trials:
            print(
                f"\nTrial {trial} already exists in "
                f"hpo_trials.csv or its individual JSON file."
            )
            print("Skipping trial.")
            return

        # Reconstruct the exact configuration for this trial.
        # This makes the configuration independent of execution order.
        trial_rng = np.random.RandomState(SEED)

        config = None

        for _ in range(trial):
            config = sample_config(trial_rng)

        print(f"\nRunning trial {trial}/{NUM_TRIALS}")
        print(json.dumps(config, indent=2))

        val_metrics = run_training(
            config,
            train_small_ds,
            val_ds,
            seed=SEED + trial
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
            f"\nTrial {trial} completed:"
            f"\nValidation MAE: {val_metrics['mae']:.6f}"
            f"\nValidation RMSE: {val_metrics['rmse']:.6f}"
        )

        print(
            f"Saved: "
            f"{result_dir / 'trials' / f'trial_{trial:03d}.json'}"
        )

        # Important: no final training in single-trial mode.
        return

    # --------------------------------------------------------
    # Normal HPO mode: process all missing trials
    # --------------------------------------------------------

    for trial in range(1, NUM_TRIALS + 1):

        if trial in completed_trials:
            print(
                f"\nTrial {trial}/{NUM_TRIALS} already completed. "
                f"Skipping."
            )
            continue

        # Reconstruct the exact configuration for this trial.
        trial_rng = np.random.RandomState(SEED)

        config = None

        for _ in range(trial):
            config = sample_config(trial_rng)

        print(f"\nTrial {trial}/{NUM_TRIALS}")
        print(json.dumps(config, indent=2))

        val_metrics = run_training(
            config,
            train_small_ds,
            val_ds,
            seed=SEED + trial
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
            flush=True
        )

    # --------------------------------------------------------
    # Verify all trials are available
    # --------------------------------------------------------

    missing_trials = [
        trial
        for trial in range(1, NUM_TRIALS + 1)
        if trial not in trial_results
    ]

    if missing_trials:
        raise RuntimeError(
            f"Cannot start final training. "
            f"Missing trials: {missing_trials}"
        )

    # Sort by trial number.
    trial_results_list = [
        trial_results[trial]
        for trial in range(1, NUM_TRIALS + 1)
    ]

    # Write the central CSV only after all trials are available.
    trials_df = pd.DataFrame(trial_results_list)

    trials_df.to_csv(
        result_dir / "hpo_trials.csv",
        index=False
    )

    print(
        f"\nAll {NUM_TRIALS} trials are available."
    )

    # --------------------------------------------------------
    # Select best configuration
    # --------------------------------------------------------

    best_result = min(
        trial_results_list,
        key=lambda result: float(result["val_rmse"])
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

    with (result_dir / "best_config.json").open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "dataset": name,
                "best_trial": int(best_result["trial"]),
                "val_mae": float(best_result["val_mae"]),
                "val_rmse": float(best_result["val_rmse"]),
                "config": best_config,
            },
            f,
            indent=2
        )

    print("\nBest configuration:")
    print(json.dumps(best_config, indent=2))

    print(
        f"Best trial: {best_result['trial']}"
        f"\nBest validation MAE: {best_result['val_mae']:.6f}"
        f"\nBest validation RMSE: {best_result['val_rmse']:.6f}"
    )

    # --------------------------------------------------------
    # Final runs: train on train_uni, evaluate on test_uni
    # --------------------------------------------------------

    final_rows = []

    for seed in FINAL_SEEDS:

        print(f"\nFinal full-training run with seed {seed}")

        metrics = run_training(
            best_config,
            train_full_ds,
            test_ds,
            seed=seed
        )

        final_rows.append({
            "dataset": name,
            "seed": seed,
            "test_mae": metrics["mae"],
            "test_rmse": metrics["rmse"],
            "best_trial": int(best_result["trial"]),
            "val_mae": float(best_result["val_mae"]),
        })

        pd.DataFrame(final_rows).to_csv(
            result_dir / "final_results_seeds.csv",
            index=False
        )

        print(
            f"Test MAE: {metrics['mae']:.6f} | "
            f"Test RMSE: {metrics['rmse']:.6f}"
        )

    seed_df = pd.DataFrame(final_rows)

    avg_row = {
        "dataset": name,
        "seed": "avg",
        "test_mae": seed_df["test_mae"].mean(),
        "test_rmse": seed_df["test_rmse"].mean(),
        "test_mae_std": seed_df["test_mae"].std(ddof=1),
        "test_rmse_std": seed_df["test_rmse"].std(ddof=1),
        "best_trial": int(best_result["trial"]),
        "val_mae": float(best_result["val_mae"]),
    }

    final_df = pd.concat(
        [seed_df, pd.DataFrame([avg_row])],
        ignore_index=True
    )

    final_df.to_csv(
        result_dir / "final_results.csv",
        index=False
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
            f"No dataset folders containing all four required "
            f"Arrow files found under {DATA_ROOT}"
        )

    print(f"Device: {DEVICE}")
    print(f"Found {len(datasets)} datasets:")

    for idx, info in enumerate(datasets):
        print(f"  [{idx}] {info['name']} — {info['dir']}")

    # Arguments:
    # python tser_bolt_hpo.py [dataset_index] [trial_index]

    if len(sys.argv) > 3:
        raise SystemExit(
            "Usage: python tser_bolt_hpo.py "
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_info in datasets:
        run_dataset(
            dataset_info,
            config_index=config_index
        )



if __name__ == "__main__":
    main()
