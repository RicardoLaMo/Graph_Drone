#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: $0 [smoke|full] [--print-only]" \
    "" \
    "Launches the current portfolio plan:" \
    "- lane A: 4 small datasets on GPUs 3,2,1,0" \
    "- lane B: diamonds wide run on GPUs 7,6,5,4"
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
cd "$repo_root"

mode="full"
print_only=0

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

lane_a_script="$repo_root/experiments/openml_regression_benchmark/scripts/run_lane_a_small_datasets.sh"
lane_b_script="$repo_root/experiments/openml_regression_benchmark/scripts/run_lane_b_diamonds.sh"

cmd_a=("$lane_a_script" "$mode")
cmd_b=("$lane_b_script" "$mode")
if [[ "$print_only" -eq 1 ]]; then
  cmd_a+=(--print-only)
  cmd_b+=(--print-only)
fi

printf '[portfolio] lane-a GPUs=%s\n' "${GRAPHDRONE_LANE_A_GPUS:-3,2,1,0}"
printf '[portfolio] lane-b GPUs=%s\n' "${GRAPHDRONE_LANE_B_GPUS:-7,6,5,4}"

if [[ "$print_only" -eq 1 ]]; then
  "${cmd_a[@]}"
  "${cmd_b[@]}"
  exit 0
fi

cleanup() {
  local pid
  for pid in "${lane_pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup INT TERM

declare -a lane_pids=()

"${cmd_a[@]}" &
lane_pids+=("$!")

"${cmd_b[@]}" &
lane_pids+=("$!")

failures=0
for pid in "${lane_pids[@]}"; do
  if ! wait "$pid"; then
    failures=1
  fi
done

exit "$failures"
