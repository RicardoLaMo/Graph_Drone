#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEEDS_TEXT="${P0_SEEDS:-41 42 43}"
SPLIT_SEED="${P0_SPLIT_SEED:-42}"
OPENML_DATASET_ID="${P0_OPENML_DATASET_ID:-44024}"
GPU_GROUPS_TEXT="${P0_GPU_GROUPS:-7,6,5,4;3,2,1,0}"
OUTPUT_ROOT="${P0_OUTPUT_ROOT:-experiments/tabpfn_view_router/reports}"
SMOKE_FLAG="${P0_SMOKE:-0}"
N_ESTIMATORS="${P0_N_ESTIMATORS:-1}"

IFS=' ' read -r -a SEEDS <<< "$SEEDS_TEXT"
IFS=';' read -r -a GPU_GROUPS <<< "$GPU_GROUPS_TEXT"

if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  echo "No seeds configured. Set P0_SEEDS." >&2
  exit 1
fi

if [[ "${#GPU_GROUPS[@]}" -eq 0 ]]; then
  echo "No GPU groups configured. Set P0_GPU_GROUPS." >&2
  exit 1
fi

build_device_list() {
  local group_csv="$1"
  local count=0
  local device_list=()
  IFS=',' read -r -a ids <<< "$group_csv"
  count="${#ids[@]}"
  if (( count > 4 )); then
    count=4
  fi
  for ((idx=0; idx<count; idx++)); do
    device_list+=("cuda:${idx}")
  done
  local joined
  joined="$(IFS=,; echo "${device_list[*]}")"
  printf '%s' "$joined"
}

launch_seed() {
  local seed="$1"
  local gpu_group="$2"
  local device_list
  device_list="$(build_device_list "$gpu_group")"
  local output_dir="${OUTPUT_ROOT}/p0_openml_seed${seed}_h200"
  local -a cmd=(
    python experiments/tabpfn_view_router/scripts/run_experiment.py
    --seed "$seed"
    --split-seed "$SPLIT_SEED"
    --n-estimators "$N_ESTIMATORS"
    --dataset-source openml
    --openml-dataset-id "$OPENML_DATASET_ID"
    --device "$device_list"
    --parallel-workers 4
    --output "$output_dir"
  )
  if [[ "$SMOKE_FLAG" == "1" ]]; then
    cmd+=(--smoke)
  fi

  (
    export CUDA_VISIBLE_DEVICES="$gpu_group"
    source migration/activate_h200_env.sh >/dev/null
    unset HF_HUB_OFFLINE
    echo "[p0-openml] seed=${seed} GPUs=${gpu_group} devices=${device_list} output=${output_dir}"
    "${cmd[@]}"
  ) &
}

echo "[p0-openml] seeds=${SEEDS_TEXT}"
echo "[p0-openml] gpu_groups=${GPU_GROUPS_TEXT}"
echo "[p0-openml] split_seed=${SPLIT_SEED} openml_did=${OPENML_DATASET_ID}"

group_count="${#GPU_GROUPS[@]}"
for ((offset=0; offset<${#SEEDS[@]}; offset+=group_count)); do
  pids=()
  batch_end=$((offset + group_count))
  if (( batch_end > ${#SEEDS[@]} )); then
    batch_end="${#SEEDS[@]}"
  fi
  for ((batch_idx=offset; batch_idx<batch_end; batch_idx++)); do
    group_idx=$((batch_idx - offset))
    launch_seed "${SEEDS[$batch_idx]}" "${GPU_GROUPS[$group_idx]}"
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done

echo "[p0-openml] complete"
