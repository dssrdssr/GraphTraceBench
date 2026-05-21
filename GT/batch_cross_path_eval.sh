#!/usr/bin/env bash
set -euo pipefail

# Batch cross-path evaluation
# Example:
#   bash batch_cross_path_eval.sh
# Optional env overrides:
#   MODELS="graphgps exphormer" EPOCH=100 SPLIT=all BATCH_SIZE=128 \
#   DATA_DIR=data RUNS_DIR=runs RESULTS_DIR=results/cross_path \
#   EVAL_SCRIPT=eval_checkpoint.py \
#   bash batch_cross_path_eval.sh

MODELS_STR="${MODELS:-graphgps}"
EPOCH="${EPOCH:-100}"
SPLIT="${SPLIT:-all}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DATA_DIR="${DATA_DIR:-data}"
RUNS_DIR="${RUNS_DIR:-runs}"
RESULTS_DIR="${RESULTS_DIR:-results/cross_path}"
EVAL_SCRIPT="${EVAL_SCRIPT:-eval_checkpoint.py}"

# Available GPUs
GPUS_STR="${GPUS:-1 2 3 4 5 6 7}"

# # Train/test path ranges
TRAIN_PATHS_STR="${TRAIN_PATHS:-1 2 3 4 5 6}"
TEST_PATHS_STR="${TEST_PATHS:-1 2 3 4 5 6}"

# BFS Train/test path ranges
# TRAIN_PATHS_STR="${TRAIN_PATHS:-1 2 3 4 5 }"
# TEST_PATHS_STR="${TEST_PATHS:-1 2 3 4 5 }"

# shuffled Train/test path ranges
# TRAIN_PATHS_STR="${TRAIN_PATHS:-3 }"
# TEST_PATHS_STR="${TEST_PATHS:-3 }"



IFS=' ' read -r -a MODELS <<< "$MODELS_STR"
IFS=' ' read -r -a GPUS <<< "$GPUS_STR"
IFS=' ' read -r -a TRAIN_PATHS <<< "$TRAIN_PATHS_STR"
IFS=' ' read -r -a TEST_PATHS <<< "$TEST_PATHS_STR"

if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "No GPUs configured."
  exit 1
fi

mkdir -p "$RESULTS_DIR/logs"

run_one() {
  local gpu="$1"
  local model="$2"
  local train_path="$3"
  local test_path="$4"

  local checkpoint="${RUNS_DIR}/${model}/path_${train_path}_epoch_${EPOCH}.pt"
  local dataset="${DATA_DIR}/path_${test_path}.pt"
  local out_dir="${RESULTS_DIR}/${model}/train_path_${train_path}"
  local output="${out_dir}/test_path_${test_path}.json"
  local log_file="${RESULTS_DIR}/logs/${model}_train${train_path}_test${test_path}.log"

  mkdir -p "$out_dir"

  if [[ ! -f "$checkpoint" ]]; then
    echo "[SKIP] Missing checkpoint: $checkpoint" | tee -a "$log_file"
    return 0
  fi

  if [[ ! -f "$dataset" ]]; then
    echo "[SKIP] Missing dataset: $dataset" | tee -a "$log_file"
    return 0
  fi

  echo "[RUN ] gpu=${gpu} model=${model} train_path=${train_path} test_path=${test_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "$EVAL_SCRIPT" \
    --checkpoint "$checkpoint" \
    --dataset "$dataset" \
    --split "$SPLIT" \
    --batch-size "$BATCH_SIZE" \
    --device cuda \
    --output "$output" > "$log_file" 2>&1

  echo "[DONE] gpu=${gpu} model=${model} train_path=${train_path} test_path=${test_path}"
}

job_count=0
for model in "${MODELS[@]}"; do
  for train_path in "${TRAIN_PATHS[@]}"; do
    for test_path in "${TEST_PATHS[@]}"; do
      gpu="${GPUS[$((job_count % ${#GPUS[@]}))]}"
      run_one "$gpu" "$model" "$train_path" "$test_path" &
      ((job_count+=1))

      # Limit concurrent jobs to number of GPUs
      if (( job_count % ${#GPUS[@]} == 0 )); then
        wait
      fi
    done
  done
done

wait
echo "All evaluations finished. Results saved under: $RESULTS_DIR"
