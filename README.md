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


## Pretraining

To pretrain the two default models used for downstream evaluation install [Chronos](https://github.com/amazon-science/chronos-forecasting) and replace the train.py file with our modified version. Furthermore replace the chronos.py or chronos_bolt.py with our version which support mlm models.


To download the pretraining data use:

```bash
python3 data/get_train_data.py
```

To run the pretraining use:

```bash
python3 src/train.py --config BertModel/configs/ChronosBERT.yaml
python3 src/train.py --config BertModel/configs/ChronosBERT-Bolt.yaml

```

The pretrained models are available at huggingface:
[ChronosBERT](https://huggingface.co/ddrg/ChronosBERT-Default)
[ChronosBERT-Bolt](https://huggingface.co/ddrg/ChronosBERT-Bolt-Default)


## Data Preparation for Finetuning

To reproduce our finetuning hpo use the provided data download and data preprocessing scripts.

### Data Download


Imputation and Similarity(ArabicSpokenDigits) datasets  can be downloaded using the provided scripts.

```bash
python3 data/finetuning/Imputation/ett_data.py
python3 data/finetuning/Similarity/similarity_data.py
```

TSER datasets and UCR classification archive have to be downloaded manually.

### Data Preprocessing

To generate the fixed masks for imputation use:

```bash
python3 data/finetuning/Imputation/create_masks.py
```

To create the time series similarity pairs for the similarity experiments.

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


## Finetuning ChronosBERT

To reproduce our finetuning hpo create the databases and insert the configs:

```bash
python3 src/finetuning/imputation/db_imputation.py
python3 src/finetuning/imputation/configs_imputation.py
```

To perform finetuning hpo and evaluation use the following scripts:

For **TSER** and **imputation**, use a worker which starts the finetunnign and evaluation script where the index is an integer which defines which Config from the db is executed.

```bash
python3 /src/finetuning/tser/worker_imputation.py <index>
```

For **classification** and **similarity**, the training and evaluation scripts are combined.

```bash
python3 BERTi/src/finetuning/classification/classification.py --index <index>
```

After the hpo is finished select the best configuration with and retrain the model on the full training data:


```bash
python3 src/finetuning/imputation/final/final_db_imputation.py
python3 src/finetuning/imputation/final/final_worker_imputation.py <index>
```

For TSER, the `final` directory also contains a script for performing the multivariate experiments.

## Finetuning ChronosBERT-Bolt


To perform hyperparameter optimization for classification or similarity, run:

```bash
python3 BERTi/src/finetuning/bolt/bolt_classification.py <DATASET_INDEX>
python3 BERTi/src/finetuning/bolt/bolt_similarity.py <DATASET_INDEX>
```

This searches for all datasets and runs the full HPO procedure including testing for the dataset corresponding to the specified index.

For TSER, there are two options. Since some datasets are large, it is also possible to run a single configuration instead of performing the full HPO procedure.


```bash
python3 BERTi/src/finetuning/bolt_tser.py \
    "<DATASET_INDEX>" \
    "<TRIAL_INDEX>"

python3 BERTi/src/finetuning/multi_bolt_tser.py \
    "<DATASET_INDEX>" \
    "<TRIAL_INDEX>"
```
