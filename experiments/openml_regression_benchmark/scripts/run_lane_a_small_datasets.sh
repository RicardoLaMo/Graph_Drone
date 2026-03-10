#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: $0 [smoke|full] [--print-only]" \
    "" \
    "Runs the small-dataset lane with one dataset pinned per GPU." \
    "Default GPUs: GRAPHDRONE_LANE_A_GPUS=3,2,1,0"
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-$repo_root/.venv-h200/bin/python}"
runner="experiments/openml_regression_benchmark/scripts/run_dataset_manifest.py"
mode="full"
print_only=0
declare -a datasets=(
  airfoil_self_noise
  concrete_compressive_strength
  healthcare_insurance_expenses
  used_fiat_500
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    smoke|full)
      mode="$1"
      shift
      ;;
    --print-only)
      print_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

IFS=',' read -r -a gpu_ids <<< "${GRAPHDRONE_LANE_A_GPUS:-3,2,1,0}"
if [[ "${#gpu_ids[@]}" -lt "${#datasets[@]}" ]]; then
  printf 'Need at least %s GPUs for lane A, got %s (%s)\n' \
    "${#datasets[@]}" "${#gpu_ids[@]}" "${GRAPHDRONE_LANE_A_GPUS:-3,2,1,0}" >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_root="$repo_root/experiments/openml_regression_benchmark/reports_lane_runs/lane_a_${mode}_$timestamp"
mkdir -p "$log_root"

declare -a pids=()
declare -a pid_labels=()

cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup INT TERM

for idx in "${!datasets[@]}"; do
  dataset="${datasets[$idx]}"
  gpu="${gpu_ids[$idx]}"
  log_path="$log_root/${dataset}.log"
  cmd=(
    "$python_bin"
    "$runner"
    --dataset "$dataset"
    --gpus "$gpu"
  )
  if [[ "$mode" == "smoke" ]]; then
    cmd+=(--smoke)
  fi

  printf '[lane-a] dataset=%s gpu=%s log=%s\n' "$dataset" "$gpu" "$log_path"
  printf '%q ' "${cmd[@]}"
  printf '\n'

  if [[ "$print_only" -eq 1 ]]; then
    continue
  fi

  "${cmd[@]}" >"$log_path" 2>&1 &
  pids+=("$!")
  pid_labels+=("$dataset")
done

if [[ "$print_only" -eq 1 ]]; then
  exit 0
fi

failures=0
for idx in "${!pids[@]}"; do
  if ! wait "${pids[$idx]}"; then
    printf '[lane-a] failed dataset=%s log=%s/%s.log\n' \
      "${pid_labels[$idx]}" "$log_root" "${pid_labels[$idx]}" >&2
    failures=1
  fi
done

exit "$failures"
