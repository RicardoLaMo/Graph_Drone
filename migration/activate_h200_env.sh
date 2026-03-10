#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Run this with: source migration/activate_h200_env.sh" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${GRAPH_DRONE_H200_VENV:-$ROOT_DIR/.venv-h200}"

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Missing Python environment at $VENV_PATH" >&2
  echo "Run: bash migration/bootstrap_h200.sh" >&2
  return 1
fi

if [[ -f "$VENV_PATH/bin/activate" ]]; then
  source "$VENV_PATH/bin/activate"
else
  export PATH="$VENV_PATH/bin:$PATH"
  export VIRTUAL_ENV="$VENV_PATH"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"

export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-$ROOT_DIR/.cache/tabpfn}"
export OPENML_CACHE_DIR="${OPENML_CACHE_DIR:-$ROOT_DIR/.cache/openml}"
export GRAPH_DRONE_GPU_COUNT="${GRAPH_DRONE_GPU_COUNT:-8}"
export GRAPH_DRONE_GPU_ORDER="${GRAPH_DRONE_GPU_ORDER:-high-first}"
export GRAPHDRONE_OPENML_GRAPHDRONE_GPU_SPAN="${GRAPHDRONE_OPENML_GRAPHDRONE_GPU_SPAN:-1}"
export GRAPHDRONE_OPENML_MAX_CONCURRENT_JOBS="${GRAPHDRONE_OPENML_MAX_CONCURRENT_JOBS:-8}"
export GRAPHDRONE_OPENML_GPU_MEMORY_FREE_THRESHOLD_MIB="${GRAPHDRONE_OPENML_GPU_MEMORY_FREE_THRESHOLD_MIB:-4096}"
export GRAPHDRONE_OPENML_GPU_UTIL_FREE_THRESHOLD="${GRAPHDRONE_OPENML_GPU_UTIL_FREE_THRESHOLD:-10}"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TABPFN_MODEL_CACHE_DIR" "$OPENML_CACHE_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
  if [[ -z "${GRAPH_DRONE_GPU_POOL:-}" ]]; then
    if [[ "$GRAPH_DRONE_GPU_ORDER" == "low-first" ]]; then
      sort_args=("-n")
    else
      sort_args=("-nr")
    fi
    mapfile -t GPU_IDS < <(
      nvidia-smi --query-gpu=index --format=csv,noheader,nounits \
        | sort "${sort_args[@]}" \
        | head -n "$GRAPH_DRONE_GPU_COUNT" \
        | tr -d ' '
    )
    if [[ "${#GPU_IDS[@]}" -gt 0 ]]; then
      export GRAPH_DRONE_GPU_POOL
      GRAPH_DRONE_GPU_POOL="$(IFS=,; echo "${GPU_IDS[*]}")"
    fi
  fi
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -n "${GRAPH_DRONE_GPU_POOL:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$GRAPH_DRONE_GPU_POOL"
  fi
fi

echo "Activated $VENV_PATH"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "GRAPH_DRONE_GPU_POOL=${GRAPH_DRONE_GPU_POOL:-<unset>}"
echo "GRAPHDRONE_OPENML_GRAPHDRONE_GPU_SPAN=$GRAPHDRONE_OPENML_GRAPHDRONE_GPU_SPAN"
echo "GRAPHDRONE_OPENML_MAX_CONCURRENT_JOBS=$GRAPHDRONE_OPENML_MAX_CONCURRENT_JOBS"
echo "HF_HOME=$HF_HOME"
echo "TABPFN_MODEL_CACHE_DIR=$TABPFN_MODEL_CACHE_DIR"
