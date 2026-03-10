#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-$repo_root/.venv-h200/bin/python}"
script_path="experiments/openml_regression_benchmark/scripts/run_openml_suite.py"
dataset_args=(
  healthcare_insurance_expenses
  concrete_compressive_strength
  airfoil_self_noise
  Another-Dataset-on-used-Fiat-500
  wine_quality
  diamonds
)

mode="${1:-smoke}"
shift || true

common_args=(
  --datasets "${dataset_args[@]}"
  --graphdrone-max-train-samples 16000
  --tabpfn-max-train-samples 16000
  --gpus "${GRAPH_DRONE_GPU_POOL:-auto}"
  --gpu-order "${GRAPH_DRONE_GPU_ORDER:-high-first}"
  --graphdrone-gpu-span "${GRAPHDRONE_OPENML_GRAPHDRONE_GPU_SPAN:-1}"
  --graphdrone-parallel-workers "${GRAPHDRONE_OPENML_PARALLEL_WORKERS:-0}"
  --max-concurrent-jobs "${GRAPHDRONE_OPENML_MAX_CONCURRENT_JOBS:-8}"
  --gpu-memory-free-threshold-mib "${GRAPHDRONE_OPENML_GPU_MEMORY_FREE_THRESHOLD_MIB:-4096}"
  --gpu-util-free-threshold "${GRAPHDRONE_OPENML_GPU_UTIL_FREE_THRESHOLD:-10}"
)

case "$mode" in
  smoke)
    output_root="experiments/openml_regression_benchmark/reports_h200_extra_smoke"
    exec "$python_bin" -c "import runpy; runpy.run_path('$script_path', run_name='__main__')" \
      "${common_args[@]}" \
      --folds 0 \
      --smoke \
      --output-root "$output_root" \
      "$@"
    ;;
  full)
    output_root="experiments/openml_regression_benchmark/reports_h200_extra_full"
    exec "$python_bin" -c "import runpy; runpy.run_path('$script_path', run_name='__main__')" \
      "${common_args[@]}" \
      --folds 0 1 2 \
      --output-root "$output_root" \
      "$@"
    ;;
  *)
    printf 'Usage: %s [smoke|full] [extra args...]\n' "$0" >&2
    exit 2
    ;;
esac
