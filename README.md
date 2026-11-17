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

**Note:** The setup scripts and Python scripts contain paths to scripts, datasets, checkpoints, and databases that may need to be adapted.

The experiments were conducted using Python 3.12.3 and CUDA cuda_12.6

The implementation is based on the original [Chronos repository](https://github.com/amazon-science/chronos-forecasting).

The following files are modified:

- `chronos.py`
- `chronos_bolt.py`

**Note:** The modified files are located in `chronos_pkg/chronos/src`.


## Pretraining

To reproduce the pretraining perfrom the follwing steps:

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

### Hyperparameter Optimization

For the desired task, initialize the configuration database and insert the configurations using:

```bash
python3 src/finetuning/<TASK>/db_<TASK>.py
python3 src/finetuning/<TASK>/configs_<TASK>.py
```

`<TASK>` can be `imputation`, `classification`, `tser`, or `similarity`.

For TSER and imputation, worker scripts execute the training and evaluation procedures. The index determines which configuration from the database is executed.

```bash
python3 src/finetuning/<TASK>/worker_<TASK>.py <INDEX>
```

For classification and similarity, training and evaluation are combined in the scripts. For example, classification can be executed using:

```bash
python3 src/finetuning/classification/classification.py --index <INDEX>
```

The same procedure applies to similarity using its script.

### Final Training and Testing

After completing HPO, initialize the final evaluation database. This selects the best configuration for each dataset.

To retrain and evaluate the model using the full training data, execute:

```bash
python3 src/finetuning/<TASK>/final/final_db_<TASK>.py
python3 src/finetuning/<TASK>/final/final_worker_<TASK>.py <INDEX>
```

For TSER, the `final` directory also contains scripts for the multivariate experiments. The workers can be adapted to use run the multivariate versions.

## Finetuning ChronosBERT-Bolt


## Classificaiton and Similarity

To perform HPO and testing for classification, similarity or tser, execute:

```bash
python3 src/finetuning/bolt/bolt_<TASK>.py <DATASET_INDEX>
```

`<TASK>` can  `classification`, or `similarity`.


The supplied dataset index determines which dataset is processed. The script performs the HPO procedure and tests the best configurations for the corresponding dataset.


### TSER

Two execution modes are supported for TSER.

The first option executes a single trial for a specified dataset which can be used for large datasets.

**Univariate TSER:**

```bash
python3 src/finetuning/bolt/bolt_tser.py <DATASET_INDEX> <TRIAL_INDEX>
```

**Multivariate TSER:**

```bash
python3 src/finetuning/bolt/multi_bolt_tser.py <DATASET_INDEX> <TRIAL_INDEX>
```

The second option executes the complete HPO procedure for a dataset. To use this mode, omit the `TRIAL_INDEX` argument:

**Univariate TSER:**

```bash
python3 src/finetuning/bolt/bolt_tser.py <DATASET_INDEX>
```

**Multivariate TSER:**

```bash
python3 src/finetuning/bolt/multi_bolt_tser.py <DATASET_INDEX>
```
