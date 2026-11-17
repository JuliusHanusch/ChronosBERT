# CodeDiplomaThesis

This repository contains the code required to reproduce the experiments presented in the thesis *Enabling BERT for Time Series Tasks by Utilizing Chronos*.

The repository provides implementations for pretraining and finetuning two proposed models:

- **ChronosBERT:** Combines Chronos time series tokenization with a BERT encoder architecture.
- **ChronosBERT-Bolt:** Combines the patch-based representation of Chronos-Bolt with a BERT encoder architecture.

Both models are evaluated on four downstream tasks: time series classification, similarity learning, imputation, and time series regression (TSER).

The repository includes scripts for data preparation, pretraining, hyperparameter optimization (HPO), and final model evaluation.


## Environment Setup

The experiments were conducted on the TU Dresden Capella HPC system.

To create the virtual environment and install the required dependencies, execute:

```bash
sbatch hpc/venvs/setup_gpu_venv.sh
```

The setup script creates the GPU environment and installs the dependencies required for the experiments.

Different environments are required for MOMENT and InceptionTime due to dependency conflicts. The corresponding requirements files are provided in the `requirements` directory.

**Note:** The setup scripts and Python scripts contain paths to scripts, datasets, checkpoints, and databases that may need to be adapted to the local environment. In particular, check the configured project root directory before executing the scripts.

The experiments were developed using Python 3.12 and CUDA 12.6.

## Pretraining

The pretraining implementation is based on the original [Chronos repository](https://github.com/amazon-science/chronos-forecasting).

To pretrain the two models, clone the Chronos repository and replace the corresponding training and model files with the modified versions provided in this repository. The modifications extend the original implementation to support masked language modeling with BERT-style encoder models.

The following files need to be replaced:

- `train.py`
- `chronos.py`
- `chronos_bolt.py`

**Note:** The modified files are located in `src` and `chronos_pkg/chronos/src`.

## Pretraining

Download the pretraining data using:

```bash
python3 data/get_train_data.py
```


The pretraining configurations for both models are located in `BertModel/configs/`.

To pretrain ChronosBERT, execute:

```bash
python3 src/train.py --config BertModel/configs/ChronosBERT.yaml
```

To pretrain ChronosBERT-Bolt, execute:

```bash
python3 src/train.py --config BertModel/configs/ChronosBERT-Bolt.yaml
```

The pretrained models used for downstream evaluation are available on Hugging Face:

- [ChronosBERT](https://huggingface.co/ddrg/ChronosBERT-Default)
- [ChronosBERT-Bolt](https://huggingface.co/ddrg/ChronosBERT-Bolt-Default)


## Data Preparation for Finetuning

The following sections describe how to download and preprocess the datasets required for the downstream experiments.

### Data Download

The ETT datasets used for imputation and the ArabicSpokenDigits dataset used for similarity learning can be downloaded using the provided scripts.

**Imputation:**

```bash
python3 data/finetuning/Imputation/ett_data.py
```

**Similarity:**

```bash
python3 data/finetuning/Similarity/similarity_data.py
```

The TSER datasets and the UCR Time Series Classification Archive have to be downloaded manually. Afterwards put the data in the directory expected by the preprocesssing script or adapt he model path.

### Data Preprocessing

After downloading the datasets, execute the following scripts to prepare the data for finetuning.

**Imputation**

Generate the fixed missing-value masks used in the imputation experiments:

```bash
python3 data/finetuning/Imputation/create_masks.py
```

**Similarity**

Generate the time series pairs used for the similarity experiments:

```bash
python3 data/finetuning/Similarity/splitandpairs.py
python3 data/finetuning/UCR_extracted/similarity_pairs.py
```

**TSER**

Preprocess the downloaded TSER datasets:

```bash
python3 data/finetuning/TSER/create_tser_data.py
```

**Classification**

Preprocess the downloaded UCR datasets:

```bash
python3 data/finetuning/UCR_extracted/UCR_classification_data.py
```

## Finetuning ChronosBERT

The following sections describe how to perform HPO and final evaluation for ChronosBERT.

For the desired task, initialize the configuration database and insert the configurations using:

```bash
python3 src/finetuning/<task>/db_<task>.py
python3 src/finetuning/<task>/configs_<task>.py
```

<task> can be imputation, classification, tser, similarity


For TSER and imputation, worker scripts execute the training and evaluation procedures for individual configurations. The supplied index determines which configuration from the corresponding database is executed.


```bash
python3 src/finetuning/tser/worker_tser.py <INDEX>
```
The same scrip applies to imputation using its corresponding script.

For classification and similarity, training and evaluation are combined in the respective scripts. For example, classification can be executed using:

```bash
python3 src/finetuning/classification/classification.py --index <INDEX>
```

The same procedure applies to similarity using its corresponding script.

### Final Evaluation

After completing HPO, inititalize the final db which selects the best configuration for each dataset and retrain the model using the full training data.

For <task>, initialize the final evaluation database and execute the corresponding worker:

```bash
python3 src/finetuning/<task>/final/final_db_<task>.py
python3 src/finetuning/<task>/final/final_worker_<task>.py <INDEX>
```

For TSER, there are also scripts for the multivariate experiments.


## Finetuning ChronosBERT-Bolt

### Classification

To perform HPO and evaluation for classification, execute:

```bash
python3 src/finetuning/bolt/bolt_classification.py <DATASET_INDEX>
```

The supplied dataset index determines which dataset is processed. The script performs the HPO procedure and evaluates the selected configurations for the corresponding dataset.

### Similarity

To perform HPO and evaluation for similarity, execute:

```bash
python3 src/finetuning/bolt/bolt_similarity.py <DATASET_INDEX>
```

The dataset index determines which dataset is processed.

### TSER

For TSER, two execution modes are supported.

The first option executes a single configuration for a specified dataset and trial. This is useful for distributing individual trials across multiple jobs, particularly for larger datasets.

**Univariate TSER:**

```bash
python3 src/finetuning/bolt/bolt_tser.py <DATASET_INDEX> <TRIAL_INDEX>
```

**Multivariate TSER:**

```bash
python3 src/finetuning/bolt/multi_bolt_tser.py <DATASET_INDEX> <TRIAL_INDEX>
```

The second option is to execute the complete HPO procedure for a dataset.
