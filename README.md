# CodeDiplomaThesis

This repository contains the code required to reproduce the results of my diploma thesis.

The repository is structured into several main directories. The `src` directory contains the training, evaluation, and baseline scripts used in the experiments.

## Environment

To create a wokring virtual environment and install requirements execute the:

```bash
sbatch hpc/venvs/setup_gpu_venv.sh
```

For MOMENT and InceptionTime different environments are requirend due to dependency conflicts. The requirements files are available in the requirements folder.

In this scrip as well as in all other python scripts paths to scripts or databases have to be adapted.

## Data Directory

The `data` directory contains the scripts and data preparation required for the experiments.

### Data Download

To download and transform the pretraining data to arrow format use:

```bash
python3 data/get_train_data.py
```

ETT, and Similarity(ArabicSpokenDigits) can be downloaded using the provided scripts.

```bash
python3 data/finetuning/Imputation/ett_data.py
python3 data/finetuning/Similarity/similarity_data.py
```


TSER datasets and UCR classification archive have to be downloaded manually.

### Data Preprocessing

The directory contains scripts for the required data preparation, including:

Generation of masks for val/test set of the imputation experiments.

```bash
python3 data/finetuning/Imputation/create_masks.py
```

Creation of similarity time series pairs for the similarity experiments.

```bash
python3 data/finetuning/Similarity/splitandpairs.py
python3 data/finetuning/UCR_extracted/similarity_pairs.py
```

Preprocess the TSER data after downloading

```bash
python3 data/finetuning/TSER/create_tser_data.py
```
Preprocess the classification data after downloading

```bash
python3 data/finetuning/UCR_extracted/UCR_classification_data.py
```


## SRC Directory

The `src` directory contains the training and evaluation scripts required for the experiments.

### Pretraining

The pretraining implementation is based on the code provided by [AION](https://github.com/JP-SystemsX/AION) and was adapted for ChronosBERT. To reproduce the results download the repository incorporate the changes.

The run the pretraining run:

```bash
python3 src/hpo.py --config src/search_configs/search.yml
```

The pretrained model is available [here](https://huggingface.co/juliushanusch/ChronosBERT-Default)


### Finetuning

The finetuning scripts are organized by task. Each task has a directory containing similar scripts. They include scripts for creating databases and generating configs from the seach space.

To create a database and insert the configs for e.g. imputation tasks run:

```bash
python3 src/finetuning/imputation/db_imputation.py
python3 src/finetuning/imputation/configs_imputation.py
```

To perform finetuning and evaluation use the following scripts:

For **TSER** and **imputation**, use a worker which starts the finetunnign and evaluation script where the index is an integer which defines which ID from the db is executed.

```bash
python3 /src/finetuning/tser/worker_imputation.py <index>
```

For **classification** and **similarity**, the training and evaluation scripts are combined.

```bash
python3 BERTi/src/finetuning/classification/classification.py --index <index>
```

### Final Experiments

Each finetuning task contains a `final` directory with the scripts used to perform the final training and testing using the best configurations identified during hyperparameter optimization. Use the final worker to perform the retraining on full training data and get test results. E.g. for imputation:

```bash
python3 src/finetuning/imputation/final/final_db_imputation.py
python3 src/finetuning/imputation/final/final_worker_imputation.py <index>
```

For TSER, the `final` directory also contains a script for performing the multivariate experiments.

### BOLT

The scripts for running the Chronos-Bolt experiments are located in the `bolt` directory.



To perform hyperparameter optimization for classification and similarity, run:

```bash
python3 BERTi/src/finetuning/bolt/bolt_classification.py <index>
python3 BERTi/src/finetuning/bolt/bolt_similarity.py <index>
```

This runs the full HPO procedure for the dataset corresponding to the specified index.

For TSER, there are two options. Since some datasets are large, it is also possible to run a single configuration instead of performing the full HPO procedure.


```bash
python3 BERTi/src/finetuning/bolt_tser.py \
    "<DATASET_INDEX>" \
    "<TRIAL_INDEX>"

python3 BERTi/src/finetuning/multi_bolt_tser.py \
    "<DATASET_INDEX>" \
    "<TRIAL_INDEX>"
```

To run full hpo or after running each config run the final training do:

```bash
python3 BERTi/src/finetuning/bolt/bolt_tser.py <index>
python3 BERTi/src/finetuning/bolt/multi_bolt_tser.py <index>
```

Both options perform HPO for the specified dataset, either by running an individual trial or by executing the complete search.

### Baselines

The repository includes classical and naive baselines as well as more advanced models for each task. Each of the following scripts searches for finetuning datasets. The index thus defines which dataset is perfromed.

Classical and naive baselines:

```bash
python3 src/finetuning/baselines/classification_baseline.py`
python3 src/finetuning/baselines/similarity_baselines.py`
python3 src/finetuning/baselines/imputation_baselines.py`
python3 src/finetuning/baselines/tser_baselines.py` --index <index>
```

Advanced baselines include:

For Classification:

```bash
python3 src/finetuning/baselines/moment_classification.py
```

For Similarity:

```bash
python3 src/finetuning/baselines/siamese_rnn.py
python3 src/finetuning/baselines/siamese_rnn_hpo.py --index <index>
```

For Imputation:

```bash
python3 src/finetuning/baselines/moment_imputation.py`
```

For TSER:

```bash
python3 src/finetuning/baselines/InceptionTime.py` --index <index>
python3 src/finetuning/baselines/fcn.py` --index <index>
python3 src/finetuning/baselines/rocket.py` --index <index>
```


## Chronos Directory

The `chronos_pkg` directory contains the modified Chronos implementation used for ChronosBERT and the downstream time series tasks. The implementation is based on the original [Chronos](https://github.com/amazon-science/chronos-forecasting) implementation. To reproduce the results incorporate the changes is the chronos package.

The main modifications in `chronos_pkg` are:

* `chronos.py` contains the modifications required to load the appropriate model head for the respective task and the additional handling of the masking token required for BERT-style masked pretraining.
* `chronos_task.py` contains the implementations of the task-specific heads used for the different downstream time series tasks.
* The TSER implementation contains both the univariate and multivariate variants in a single file.
