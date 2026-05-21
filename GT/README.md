# Arithmetic Graph Model Zoo

This repository provides a unified model zoo for the arithmetic-path graph regression task. It includes dataset generation scripts, training and evaluation pipelines, batch cross-path evaluation, and result aggregation utilities.

## 1. Environment Setup

Install all required dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

Make sure your CUDA environment is properly configured if you plan to train or evaluate models on GPU.

---

## 2. Model Zoo

The following model names are supported:

- `graphgps`
- `exphormer`
- `graphormer`
- `grit`
- `difformer`
- `tigt`
- `patchgt`
- `graphgpt`
- `g2pt`

These models can be passed to the training and evaluation scripts through the `--models` argument.

---

## 3. Generate Datasets

This package supports generating arithmetic-path graph datasets for different settings.

### 3.1 Generate a GraphLR-Path Dataset

Use `generate_dataset.py` to generate a GraphLR-Path dataset:

```bash
python generate_dataset.py \
  --output data/path_5.pt \
  --num-graphs 5000 \
  --num-nodes 24 \
  --path-len 5 \
  --seed 42 \
  --ops '+ - * /'
```

### 3.2 Generate a GraphLR-BFS Dataset

Use `generation_BFS.py` to generate a GraphLR-BFS dataset:

```bash
python generation_BFS.py \
  --output data/path_5.pt \
  --num-graphs 5000 \
  --num-nodes 24 \
  --path-len 5 \
  --seed 42 \
  --ops '+ - * /'
```

### Main Arguments

- `--output`: Path where the generated `.pt` dataset will be saved.
- `--num-graphs`: Number of graphs to generate.
- `--num-nodes`: Number of nodes in each graph.
- `--path-len`: Arithmetic path length.
- `--seed`: Random seed for reproducibility.
- `--ops`: Arithmetic operations used in the dataset.

---

## 4. Train Models

Use `train_eval.py` to train one or more models on a generated dataset.

Example command:

```bash
CUDA_VISIBLE_DEVICES=0 python train_eval.py \
  --dataset data_md/path_1.pt \
  --models graphgps exphormer graphormer grit difformer tigt patchgt graphgpt g2pt \
  --output-dir runs_shuffled \
  --epochs 100 \
  --batch-size 128 \
  --device cuda \
  --amp
```

### Main Arguments

- `--dataset`: Path to the `.pt` dataset file.
- `--models`: List of model names to train.
- `--output-dir`: Directory where training checkpoints and logs will be saved.
- `--epochs`: Number of training epochs.
- `--batch-size`: Training batch size.
- `--device`: Device used for training, such as `cuda` or `cpu`.
- `--amp`: Enables automatic mixed precision training.

---

## 5. Evaluate a Checkpoint on a Test Set

Use `eval_checkpoint.py` to evaluate a trained checkpoint on a specific dataset split.

Example command:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_checkpoint.py \
  --checkpoint runs_shuffled/100/graphgps/path_3_epoch_100.pt \
  --dataset data_shuffled/path_3_100.pt \
  --split all \
  --batch-size 128 \
  --device cuda \
  --output results/results_shuffled/graphgps/test_path_3_100.json
```

### Main Arguments

- `--checkpoint`: Path to the trained model checkpoint.
- `--dataset`: Path to the test dataset.
- `--split`: Dataset split to evaluate. Use `all` to evaluate the full dataset.
- `--batch-size`: Evaluation batch size.
- `--device`: Device used for evaluation.
- `--output`: Path where the evaluation results will be saved.

---

## 6. Batch Cross-Path Evaluation

Use `batch_cross_path_eval.sh` to evaluate multiple models across multiple path settings.

Example command:

```bash
DATA_DIR=data_BFS_as \
RUNS_DIR=runs_BFS_as \
RESULTS_DIR=results/results_BFS_as/cross_path \
MODELS="graphgps exphormer graphormer grit difformer tigt patchgt graphgpt g2pt" \
bash batch_cross_path_eval.sh
```

The detailed evaluation parameters are configured inside `batch_cross_path_eval.sh`.

### Environment Variables

- `DATA_DIR`: Directory containing the datasets used for evaluation.
- `RUNS_DIR`: Directory containing trained model checkpoints.
- `RESULTS_DIR`: Directory where batch evaluation results will be saved.
- `MODELS`: Space-separated list of models to evaluate.

---

## 7. Aggregate Results

Use `aggregate_cross_path_results.py` to merge cross-path evaluation results and generate summary tables.

Example command:

```bash
python aggregate_cross_path_results.py \
  --runs-dir ./runs_BFS_allop \
  --cross-dir ./results/results_BFS_allop/cross_path \
  --output-dir ./results/results_BFS_allop/tables \
  --models graphgps exphormer graphormer grit difformer tigt patchgt graphgpt g2pt \
  --epoch 100
```

### Main Arguments

- `--runs-dir`: Directory containing training outputs and checkpoints.
- `--cross-dir`: Directory containing cross-path evaluation results.
- `--output-dir`: Directory where aggregated tables will be saved.
- `--models`: List of model names to include in the aggregation.
- `--epoch`: Checkpoint epoch used for aggregation.

---

## Typical Workflow

A typical workflow is:

1. Install dependencies.
2. Generate a GraphLR-Path or GraphLR-BFS dataset.
3. Train one or more models from the model zoo.
4. Evaluate checkpoints on target test sets.
5. Run batch cross-path evaluation if needed.
6. Aggregate evaluation results into summary tables.

Example:

```bash
pip install -r requirements.txt

python generate_dataset.py \
  --output data/path_5.pt \
  --num-graphs 5000 \
  --num-nodes 24 \
  --path-len 5 \
  --seed 42 \
  --ops '+ - * /'

CUDA_VISIBLE_DEVICES=0 python train_eval.py \
  --dataset data_md/path_1.pt \
  --models graphgps exphormer graphormer grit difformer tigt patchgt graphgpt g2pt \
  --output-dir runs_shuffled \
  --epochs 100 \
  --batch-size 128 \
  --device cuda \
  --amp

CUDA_VISIBLE_DEVICES=0 python eval_checkpoint.py \
  --checkpoint runs_shuffled/100/graphgps/path_3_epoch_100.pt \
  --dataset data_shuffled/path_3_100.pt \
  --split all \
  --batch-size 128 \
  --device cuda \
  --output results/results_shuffled/graphgps/test_path_3_100.json
```

---

## Notes

- The training and evaluation commands assume that CUDA is available.
- Use `CUDA_VISIBLE_DEVICES=0` to run on GPU `0`.
- If evaluating on CPU, set `--device cpu` and remove `CUDA_VISIBLE_DEVICES=0`.
- Make sure the dataset paths, checkpoint paths, and output directories match your local project structure.
