# Graph Reasoning LLM/VLM Pipeline

This repository provides scripts for converting `.pt` graph reasoning datasets into JSONL prompt files, fine-tuning Qwen-VL models with SFT, and evaluating both local Hugging Face models and closed-source API-based models.

## 1. Environment Setup

Install the required dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

Make sure your CUDA environment is properly configured if you plan to train or evaluate models on GPU.

---

## 2. Convert `.pt` Datasets to JSONL Prompts

The script `serialize_graph_reasoning_to_jsonl.py` is used to convert `.pt` graph reasoning datasets into JSONL files for evaluation or supervised fine-tuning.

### 2.1 Convert a Test Set for Evaluation Only

Use this command when you only need an evaluation dataset:

```bash
python serialize_graph_reasoning_to_jsonl.py \
  --input rq2/data_md/path_1.pt \
  --output rq2/data_llm_md/path_eval_1.jsonl \
  --split test \
  --mode eval \
  --style compact \
  --include-target
```

### 2.2 Convert a Training Set for SFT

Use this command to create a training dataset for supervised fine-tuning:

```bash
python serialize_graph_reasoning_to_jsonl.py \
  --input rq2/data_md/path_1.pt \
  --output rq2/data_llm_md/path_1.jsonl \
  --split train \
  --mode sft_chat \
  --style compact \
  --include-system
```

### 2.3 Convert a Test Set for SFT Evaluation

Use this command to create a test dataset in SFT chat format:

```bash
python serialize_graph_reasoning_to_jsonl.py \
  --input rq2/data_md/path_1.pt \
  --output rq2/data_llm_md/path_1_test.jsonl \
  --split test \
  --mode sft_chat \
  --style compact \
  --include-system
```

---

## 3. Download Models Locally

Download the required Qwen vision-language model from Hugging Face.

For example, download either:

- `Qwen2-VL`
- `Qwen3-VL`

Place the downloaded model under the local `model/` directory, for example:

```text
./model/Qwen2-VL-2B-Instruct
```

---

## 4. Train a Qwen Model

The script `train_qwen_vl_text_sft.py` can be used to fine-tune a local Qwen-VL model with supervised fine-tuning.

Example command:

```bash
CUDA_VISIBLE_DEVICES=0 python train_qwen_vl_text_sft.py \
  --model_name ./model/Qwen2-VL-2B-Instruct \
  --train_file data_llm_shuffled/path_3_005.jsonl \
  --eval_file data_llm_shuffled/path_3_005_test.jsonl \
  --output_dir outputs_sft_shuffled/qwen2_path_3_005_sft \
  --use_4bit \
  --bf16 \
  --gradient_checkpointing
```

### Main Arguments

- `--model_name`: Path to the local base model.
- `--train_file`: JSONL file used for SFT training.
- `--eval_file`: JSONL file used for evaluation during training.
- `--output_dir`: Directory where the fine-tuned model or adapter will be saved.
- `--use_4bit`: Enables 4-bit quantization to reduce GPU memory usage.
- `--bf16`: Uses bfloat16 precision.
- `--gradient_checkpointing`: Enables gradient checkpointing to reduce memory usage during training.

---

## 5. Evaluate a Local Hugging Face Model

Use `eval_local_hf_model.py` to evaluate a local Hugging Face model, optionally with a fine-tuned adapter.

Example command:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_local_hf_model.py \
  --model_path ./model/Qwen2-VL-2B-Instruct \
  --adapter_path ./rq2/outputs_sft_as/qwen2_path_6_sft \
  --is_vl \
  --eval_file rq2/data_llm_as/path_eval_6.jsonl \
  --train_file rq2/data_llm_as/path_6.jsonl \
  --few_shot_k 0 \
  --output_file rq2/results/results_llm_as/qwen2vl/qwen2vl_path6_zero_local.json
```

### Main Arguments

- `--model_path`: Path to the local base model.
- `--adapter_path`: Path to the fine-tuned adapter. Remove this argument if evaluating the base model only.
- `--is_vl`: Indicates that the model is a vision-language model.
- `--eval_file`: JSONL evaluation file.
- `--train_file`: JSONL training file used for few-shot sampling if `few_shot_k > 0`.
- `--few_shot_k`: Number of few-shot examples. Use `0` for zero-shot evaluation.
- `--output_file`: Path where the evaluation results will be saved.

---

## 6. Evaluate Closed-Source Models

Use `eval_closed_models.py` to evaluate closed-source models through an OpenAI-compatible API.

First, set the API environment variables:

```bash
export OPENAI_API_KEY=API_KEY
export OPENAI_BASE_URL=BASE_URL
```

Then run the evaluation script:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_closed_models.py \
  --provider openai \
  --model gpt-5.4 \
  --eval_file data_llm_BFS_allop/path_eval_2.jsonl \
  --train_file data_llm_BFS_allop/path_2.jsonl \
  --few_shot_k 0 \
  --output_file results_llm_BFS_allop/gpt-54/path_2_zeroshot.json \
  --openai-base-url "BASE_URL" \
  --max_samples 100 \
  --resume
```

### Main Arguments

- `--provider`: API provider name. Use `openai` for OpenAI-compatible APIs.
- `--model`: Model name used for evaluation.
- `--eval_file`: JSONL evaluation file.
- `--train_file`: JSONL training file used for few-shot examples if enabled.
- `--few_shot_k`: Number of few-shot examples. Use `0` for zero-shot evaluation.
- `--output_file`: Path where the evaluation results will be saved.
- `--openai-base-url`: Base URL of the OpenAI-compatible API service.
- `--max_samples`: Maximum number of evaluation samples.
- `--resume`: Resume evaluation from existing results if supported.

---

## Typical Workflow

A common workflow is:

1. Install dependencies.
2. Convert the `.pt` dataset into JSONL format.
3. Download the required Qwen model locally.
4. Fine-tune the model using SFT.
5. Evaluate the fine-tuned local model.
6. Optionally evaluate closed-source models for comparison.

Example:

```bash
pip install -r requirements.txt

python serialize_graph_reasoning_to_jsonl.py \
  --input rq2/data_md/path_1.pt \
  --output rq2/data_llm_md/path_1.jsonl \
  --split train \
  --mode sft_chat \
  --style compact \
  --include-system

CUDA_VISIBLE_DEVICES=0 python train_qwen_vl_text_sft.py \
  --model_name ./model/Qwen2-VL-2B-Instruct \
  --train_file data_llm_shuffled/path_3_005.jsonl \
  --eval_file data_llm_shuffled/path_3_005_test.jsonl \
  --output_dir outputs_sft_shuffled/qwen2_path_3_005_sft \
  --use_4bit \
  --bf16 \
  --gradient_checkpointing
```
