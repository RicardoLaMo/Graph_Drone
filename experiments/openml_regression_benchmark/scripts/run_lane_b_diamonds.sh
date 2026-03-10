#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: $0 [smoke|full] [--print-only]" \
    "" \
    "Runs the wide GraphDrone lane for diamonds on a reserved 4-GPU group." \
    "Default GPUs: GRAPHDRONE_LANE_B_GPUS=7,6,5,4"
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-$repo_root/.venv-h200/bin/python}"
runner="experiments/openml_regression_benchmark/scripts/run_dataset_manifest.py"
mode="full"
print_only=0
dataset="${GRAPHDRONE_LANE_B_DATASET:-diamonds}"
gpu_group="${GRAPHDRONE_LANE_B_GPUS:-7,6,5,4}"

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

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_root="$repo_root/experiments/openml_regression_benchmark/reports_lane_runs/lane_b_${dataset}_${mode}_$timestamp"
mkdir -p "$log_root"
log_path="$log_root/${dataset}.log"

cmd=(
  "$python_bin"
  "$runner"
  --dataset "$dataset"
  --gpus "$gpu_group"
  --exclusive-graphdrone
)
if [[ "$mode" == "smoke" ]]; then
  cmd+=(--smoke)
fi

printf '[lane-b] dataset=%s gpus=%s log=%s\n' "$dataset" "$gpu_group" "$log_path"
printf '%q ' "${cmd[@]}"
printf '\n'

if [[ "$print_only" -eq 1 ]]; then
  exit 0
fi

child_pid=""
cleanup() {
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill "$child_pid" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

"${cmd[@]}" >"$log_path" 2>&1 &
child_pid="$!"
wait "$child_pid"
