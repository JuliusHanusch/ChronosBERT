
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    "/data/horse/ws/juha972b-AION-BERT-Chronos/BERTi"
)

SIMILARITY_DIR = (
    BASE_DIR / "Results/Finetuning/Similarity"
)

CHRONOS_DB = (
    BASE_DIR
    / "src/finetuning/similarity/final/similarity_cv_best.db"
)

BASELINE_CSV = (
    SIMILARITY_DIR / "similarity_baseline.csv"
)

SRN_CSV = (
    SIMILARITY_DIR / "siamese_rnn_hpo_test_results.csv"
)

BOLT_DIR = SIMILARITY_DIR / "Bolt_HPO"

OUTPUT_CSV = (
    SIMILARITY_DIR / "similarity_all_algorithms_per_dataset.csv"
)

OUTPUT_TEX = (
    SIMILARITY_DIR / "similarity_all_algorithms_per_dataset.tex"
)


# ============================================================
# CONFIGURATION
# ============================================================

EXCLUDED_DATASETS = [
    "Fungi",
    "plaid",
    "FordA",
    "HandOutlines",
]

ALGORITHMS = [
    "ChronosBERT",
    "ChronosBERT-Bolt",
    "SRN",
    "DTW",
    "Euclidean",
    "Naive",
]

METRICS = ["accuracy", "auroc"]


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_name(value):
    return (
        str(value)
        .strip()
        .replace("\\", "/")
        .split("/")[-1]
        .removesuffix(".tsv")
        .removesuffix(".txt")
        .removesuffix(".csv")
        .strip()
        .lower()
    )


def is_excluded(dataset):
    excluded = {
        normalize_name(name)
        for name in EXCLUDED_DATASETS
    }
    return normalize_name(dataset) in excluded


def to_numeric(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )
    return df


def find_column(df, candidates):
    """Return the first matching column, ignoring case."""
    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def empty_algorithm_frame():
    return pd.DataFrame(
        columns=[
            "dataset_key",
            "dataset",
            "accuracy_mean",
            "accuracy_std",
            "auroc_mean",
            "auroc_std",
        ]
    )


def make_result_frame(
    df,
    dataset_col,
    accuracy_col=None,
    accuracy_std_col=None,
    auroc_col=None,
    auroc_std_col=None,
):
    """Standardize results from a particular algorithm."""

    if dataset_col not in df.columns:
        raise ValueError(
            f"Dataset column '{dataset_col}' is missing."
        )

    result = pd.DataFrame()

    result["dataset"] = df[dataset_col].astype(str)
    result["dataset_key"] = result["dataset"].map(
        normalize_name
    )

    result["accuracy_mean"] = (
        pd.to_numeric(df[accuracy_col], errors="coerce")
        if accuracy_col
        else np.nan
    )

    result["accuracy_std"] = (
        pd.to_numeric(
            df[accuracy_std_col], errors="coerce"
        )
        if accuracy_std_col
        else np.nan
    )

    result["auroc_mean"] = (
        pd.to_numeric(df[auroc_col], errors="coerce")
        if auroc_col
        else np.nan
    )

    result["auroc_std"] = (
        pd.to_numeric(
            df[auroc_std_col], errors="coerce"
        )
        if auroc_std_col
        else np.nan
    )

    result = result.loc[
        result["dataset_key"].ne("")
        & ~result["dataset"].map(is_excluded)
    ].copy()

    return result


# ============================================================
# CHRONOSBERT
# ============================================================

def load_chronos_results():

    print("Loading ChronosBERT results...")

    with sqlite3.connect(CHRONOS_DB) as conn:
        schema = pd.read_sql_query(
            "PRAGMA table_info(runs)", conn
        )

        available = set(schema["name"])

        # Existing database uses accuracy_avg and accuracy_std.
        accuracy_col = find_column(
            pd.DataFrame(columns=available),
            ["accuracy_avg", "accuracy_mean", "test_accuracy"],
        )
        accuracy_std_col = find_column(
            pd.DataFrame(columns=available),
            ["accuracy_std", "test_accuracy_std"],
        )

        auroc_col = find_column(
            pd.DataFrame(columns=available),
            [
                "auroc_avg",
                "auroc_mean",
                "test_auroc",
                "auroc",
            ],
        )
        auroc_std_col = find_column(
            pd.DataFrame(columns=available),
            ["auroc_std", "test_auroc_std"],
        )

        selected = ["dataset"]

        for column in [
            accuracy_col,
            accuracy_std_col,
            auroc_col,
            auroc_std_col,
        ]:
            if column and column not in selected:
                selected.append(column)

        if accuracy_col is None:
            raise ValueError(
                "No accuracy column found in ChronosBERT database. "
                f"Available columns: {sorted(available)}"
            )

        query = (
            "SELECT "
            + ", ".join(f'"{c}"' for c in selected)
            + " FROM runs"
        )

        df = pd.read_sql_query(query, conn)

    print(
        "ChronosBERT columns:",
        {
            "accuracy": accuracy_col,
            "accuracy_std": accuracy_std_col,
            "auroc": auroc_col,
            "auroc_std": auroc_std_col,
        },
    )

    result = make_result_frame(
        df,
        dataset_col="dataset",
        accuracy_col=accuracy_col,
        accuracy_std_col=accuracy_std_col,
        auroc_col=auroc_col,
        auroc_std_col=auroc_std_col,
    )

    # Average duplicate records, if any.
    result = (
        result.groupby("dataset_key", as_index=False)
        .agg(
            dataset=("dataset", "first"),
            accuracy_mean=("accuracy_mean", "mean"),
            accuracy_std=("accuracy_std", "mean"),
            auroc_mean=("auroc_mean", "mean"),
            auroc_std=("auroc_std", "mean"),
        )
    )

    print(f"ChronosBERT datasets: {len(result)}")
    return result


# ============================================================
# CHRONOSBERT-BOLT
# ============================================================

def load_bolt_results():

    print("Loading ChronosBERT-Bolt results...")

    rows = []

    if not BOLT_DIR.exists():
        raise FileNotFoundError(
            f"Bolt results directory not found: {BOLT_DIR}"
        )

    for dataset_dir in sorted(BOLT_DIR.iterdir()):

        if not dataset_dir.is_dir():
            continue

        if is_excluded(dataset_dir.name):
            continue

        manual_file = (
            dataset_dir / "final_results_manual.csv"
        )
        regular_file = (
            dataset_dir / "final_results.csv"
        )

        result_file = (
            manual_file
            if manual_file.is_file()
            else regular_file
        )

        if not result_file.is_file():
            continue

        try:
            df = pd.read_csv(result_file)
        except (OSError, pd.errors.ParserError) as exc:
            print(f"Skipping {dataset_dir.name}: {exc}")
            continue

        required = {
            "seed",
            "test_accuracy",
            "test_auroc",
        }

        if not required.issubset(df.columns):
            print(
                f"Skipping {dataset_dir.name}: "
                "missing required Bolt columns"
            )
            continue

        df = to_numeric(
            df, ["test_accuracy", "test_auroc"]
        )

        is_avg = (
            df["seed"].astype(str).str.strip().str.lower()
            == "avg"
        )

        avg_rows = df.loc[is_avg]

        if avg_rows.empty:
            print(
                f"Skipping {dataset_dir.name}: "
                "no avg row found"
            )
            continue

        avg = avg_rows.iloc[0]
        seeds = df.loc[~is_avg]

        rows.append({
            "dataset": dataset_dir.name,
            "dataset_key": normalize_name(dataset_dir.name),
            "accuracy_mean": avg["test_accuracy"],
            "accuracy_std": (
                seeds["test_accuracy"].std(ddof=1)
                if seeds["test_accuracy"].count() > 1
                else np.nan
            ),
            "auroc_mean": avg["test_auroc"],
            "auroc_std": (
                seeds["test_auroc"].std(ddof=1)
                if seeds["test_auroc"].count() > 1
                else np.nan
            ),
        })

    result = pd.DataFrame(rows)

    if result.empty:
        print("Warning: no Bolt results were loaded.")
        return empty_algorithm_frame()

    # Protect against duplicate normalized dataset names.
    if result["dataset_key"].duplicated().any():
        duplicates = result.loc[
            result["dataset_key"].duplicated(keep=False),
            "dataset",
        ].tolist()

        raise ValueError(
            f"Duplicate Bolt datasets: {duplicates}"
        )

    print(f"ChronosBERT-Bolt datasets: {len(result)}")
    return result


# ============================================================
# SRN
# ============================================================

def load_srn_results():

    print("Loading SRN results...")

    df = pd.read_csv(SRN_CSV)

    dataset_col = find_column(df, ["dataset"])
    accuracy_col = find_column(
        df, ["test_accuracy", "accuracy"]
    )
    accuracy_std_col = find_column(
        df, ["test_accuracy_std", "accuracy_std"]
    )
    auroc_col = find_column(
        df, ["test_auroc", "auroc", "test_auc", "auc"]
    )
    auroc_std_col = find_column(
        df, ["test_auroc_std", "auroc_std", "test_auc_std"]
    )

    if dataset_col is None or accuracy_col is None:
        raise ValueError(
            f"Cannot identify dataset/accuracy columns in {SRN_CSV}. "
            f"Found: {df.columns.tolist()}"
        )

    result = make_result_frame(
        df,
        dataset_col,
        accuracy_col,
        accuracy_std_col,
        auroc_col,
        auroc_std_col,
    )

    # Expect one selected SRN result per dataset.
    if result["dataset_key"].duplicated().any():
        duplicates = result.loc[
            result["dataset_key"].duplicated(keep=False),
            "dataset",
        ].tolist()

        raise ValueError(
            "Multiple SRN rows found for the same dataset. "
            "Select the intended SRN model or aggregate its seeds "
            f"before generating the table. Examples: {duplicates[:20]}"
        )

    print(
        "SRN datasets:",
        len(result),
        "| AUROC available:",
        result["auroc_mean"].notna().sum(),
    )

    return result


# ============================================================
# CLASSICAL BASELINES: DTW, EUCLIDEAN, NAIVE
# ============================================================

def load_classical_baselines():

    print("Loading classical baselines...")

    df = pd.read_csv(BASELINE_CSV)

    required = {"dataset", "method", "accuracy"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {BASELINE_CSV}: {sorted(missing)}"
        )

    df["dataset_key"] = df["dataset"].map(normalize_name)
    df["method_key"] = (
        df["method"].astype(str).str.strip().str.lower()
    )

    df = df.loc[
        ~df["dataset"].map(is_excluded)
    ].copy()

    # AUROC is optional, but is loaded if the CSV contains it.
    auroc_col = find_column(
        df, ["auroc", "test_auroc", "auc", "test_auc"]
    )

    accuracy_std_col = find_column(
        df, ["accuracy_std", "test_accuracy_std"]
    )
    auroc_std_col = find_column(
        df, ["auroc_std", "test_auroc_std", "auc_std"]
    )

    df["accuracy"] = pd.to_numeric(
        df["accuracy"], errors="coerce"
    )

    if auroc_col:
        df[auroc_col] = pd.to_numeric(
            df[auroc_col], errors="coerce"
        )

    method_map = {
        "dtw": "DTW",
        "euclidean": "Euclidean",
        "naive": "Naive",
    }

    df["algorithm"] = df["method_key"].map(method_map)
    df = df.dropna(subset=["algorithm", "accuracy"])

    aggregations = {
        "dataset": ("dataset", "first"),
        "accuracy_mean": ("accuracy", "mean"),
    }

    if accuracy_std_col:
        df[accuracy_std_col] = pd.to_numeric(
            df[accuracy_std_col], errors="coerce"
        )
        aggregations["accuracy_std"] = (
            accuracy_std_col, "mean"
        )

    if auroc_col:
        aggregations["auroc_mean"] = (auroc_col, "mean")

    if auroc_std_col:
        df[auroc_std_col] = pd.to_numeric(
            df[auroc_std_col], errors="coerce"
        )
        aggregations["auroc_std"] = (
            auroc_std_col, "mean"
        )

    grouped = (
        df.groupby(
            ["dataset_key", "algorithm"], as_index=False
        )
        .agg(**aggregations)
    )

    # Convert each algorithm to the standardized format.
    results = {}

    for algorithm in ["DTW", "Euclidean", "Naive"]:

        subset = grouped.loc[
            grouped["algorithm"] == algorithm
        ].copy()

        if "accuracy_std" not in subset.columns:
            subset["accuracy_std"] = np.nan

        if "auroc_mean" not in subset.columns:
            subset["auroc_mean"] = np.nan

        if "auroc_std" not in subset.columns:
            subset["auroc_std"] = np.nan

        # Naive has accuracy only.
        if algorithm == "Naive":
            subset["auroc_mean"] = np.nan
            subset["auroc_std"] = np.nan

        results[algorithm] = subset[
            [
                "dataset_key",
                "dataset",
                "accuracy_mean",
                "accuracy_std",
                "auroc_mean",
                "auroc_std",
            ]
        ]

        print(
            f"{algorithm} datasets: {len(subset)} | "
            f"AUROC available: "
            f"{subset['auroc_mean'].notna().sum()}"
        )

    return results


# ============================================================
# MERGE ALL ALGORITHMS
# ============================================================

def build_combined_results():

    sources = {
        "ChronosBERT": load_chronos_results(),
        "ChronosBERT-Bolt": load_bolt_results(),
        "SRN": load_srn_results(),
    }

    sources.update(load_classical_baselines())

    # Use the union of dataset names across all algorithms.
    all_datasets = {}

    for df in sources.values():
        for _, row in df.iterrows():
            key = row["dataset_key"]

            if key not in all_datasets:
                all_datasets[key] = row["dataset"]

    combined = pd.DataFrame([
        {
            "dataset_key": key,
            "dataset": name,
        }
        for key, name in all_datasets.items()
    ])

    combined = combined.sort_values(
        "dataset", key=lambda col: col.str.lower()
    ).reset_index(drop=True)

    for algorithm, df in sources.items():

        rename_map = {
            "accuracy_mean": f"{algorithm}_accuracy_mean",
            "accuracy_std": f"{algorithm}_accuracy_std",
            "auroc_mean": f"{algorithm}_auroc_mean",
            "auroc_std": f"{algorithm}_auroc_std",
        }

        merge_df = df[
            [
                "dataset_key",
                "accuracy_mean",
                "accuracy_std",
                "auroc_mean",
                "auroc_std",
            ]
        ].rename(columns=rename_map)

        combined = combined.merge(
            merge_df,
            on="dataset_key",
            how="left",
            validate="one_to_one",
        )

    # Put dataset first, then metrics grouped by algorithm.
    ordered_columns = ["dataset"]

    for algorithm in ALGORITHMS:
        for metric in METRICS:
            ordered_columns.extend([
                f"{algorithm}_{metric}_mean",
                f"{algorithm}_{metric}_std",
            ])

    combined = combined[
        [
            column
            for column in ordered_columns
            if column in combined.columns
        ]
    ]

    return combined


# ============================================================
# LATEX FORMATTING
# ============================================================

def escape_latex(value):

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }

    text = str(value)

    # Backslash must be escaped first.
    text = text.replace("\\", replacements["\\"])

    for old in ["&", "%", "$", "#", "_", "{", "}"]:
        text = text.replace(old, replacements[old])

    return text


def format_metric(mean, std):

    if pd.isna(mean):
        return "--"

    if pd.isna(std):
        return f"{100 * mean:.2f}\\%"

    return (
        f"{100 * mean:.2f}\\% $\\pm$ "
        f"{100 * std:.2f}\\%"
    )


def format_algorithm_cell(row, algorithm):

    accuracy = format_metric(
        row[f"{algorithm}_accuracy_mean"],
        row[f"{algorithm}_accuracy_std"],
    )

    auroc = format_metric(
        row[f"{algorithm}_auroc_mean"],
        row[f"{algorithm}_auroc_std"],
    )

    return (
        r"\shortstack{"
        + r"Acc.: "
        + accuracy
        + r" \\ "
        + r"AUROC: "
        + auroc
        + "}"
    )


# ============================================================
# CREATE LATEX LONGTABLE
# ============================================================

def create_latex_table(df):

    lines = [
        r"\begin{longtable}{lrrrrrr}",
        r"\caption{Similarity classification accuracy and AUROC "
        r"across datasets for all evaluated algorithms. "
        r"Values are percentages; mean $\pm$ standard deviation "
        r"is reported where available. Naive provides accuracy only.}",
        r"\label{tab:similarity_all_algorithms_per_dataset} \\",
        r"\toprule",
        (
            r"Dataset & ChronosBERT & ChronosBERT-Bolt & "
            r"SRN & DTW & Euclidean & Naive \\"
        ),
        r"\midrule",
        r"\endfirsthead",
        "",
        r"\multicolumn{7}{c}%",
        r"{\tablename\ \thetable{} -- continued from previous page} \\",
        r"\toprule",
        (
            r"Dataset & ChronosBERT & ChronosBERT-Bolt & "
            r"SRN & DTW & Euclidean & Naive \\"
        ),
        r"\midrule",
        r"\endhead",
        "",
        r"\midrule",
        r"\multicolumn{7}{r}{Continued on next page} \\",
        r"\endfoot",
        "",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    for _, row in df.iterrows():

        cells = [
            escape_latex(row["dataset"])
        ]

        for algorithm in ALGORITHMS:
            cells.append(
                format_algorithm_cell(row, algorithm)
            )

        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        "",
        r"\end{longtable}",
        "",
    ])

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GENERATING COMBINED SIMILARITY RESULTS")
    print("=" * 70)

    combined = build_combined_results()

    # Save numeric data for further analysis.
    combined.to_csv(
        OUTPUT_CSV,
        index=False,
        float_format="%.6f",
    )

    # Generate LaTeX.
    latex = create_latex_table(combined)

    with OUTPUT_TEX.open(
        "w", encoding="utf-8"
    ) as file:
        file.write(
            "% Automatically generated similarity results.\n"
            "% Each algorithm cell contains accuracy and AUROC.\n\n"
        )
        file.write(latex)

    print("\n" + "=" * 70)
    print("FINISHED")
    print("=" * 70)

    print(f"Datasets in combined table: {len(combined)}")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"LaTeX: {OUTPUT_TEX}")

    print("\nMatched datasets and AUROC coverage:")

    for algorithm in ALGORITHMS:
        accuracy_col = f"{algorithm}_accuracy_mean"
        auroc_col = f"{algorithm}_auroc_mean"

        print(
            f"{algorithm:18s} "
            f"accuracy: {combined[accuracy_col].notna().sum():3d} | "
            f"AUROC: {combined[auroc_col].notna().sum():3d}"
        )

    print("\nMean accuracy and AUROC across available datasets:")

    for algorithm in ALGORITHMS:
        accuracy_col = f"{algorithm}_accuracy_mean"
        auroc_col = f"{algorithm}_auroc_mean"

        accuracy = combined[accuracy_col].mean()
        auroc = combined[auroc_col].mean()

        accuracy_text = (
            f"{accuracy:.4f}" if pd.notna(accuracy) else "--"
        )
        auroc_text = (
            f"{auroc:.4f}" if pd.notna(auroc) else "--"
        )

        print(
            f"{algorithm:18s} "
            f"accuracy={accuracy_text}, AUROC={auroc_text}"
        )


if __name__ == "__main__":
    main()
