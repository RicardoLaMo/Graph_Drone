#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${1:-$ROOT_DIR/.venv-h200}"
CONDA_BIN="${CONDA_BIN:-conda}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_SPEC="${TORCH_SPEC:-conda-forge::pytorch=2.6.*=cuda126*}"
TORCHVISION_SPEC="${TORCHVISION_SPEC:-conda-forge::torchvision}"
TORCHAUDIO_SPEC="${TORCHAUDIO_SPEC:-conda-forge::torchaudio}"
FAISS_SPEC="${FAISS_SPEC:-pytorch::faiss-gpu-cuvs=1.14.1=*cuda12.6*}"
LIBNVJITLINK_SPEC="${LIBNVJITLINK_SPEC:-nvidia::libnvjitlink}"
TABPFN_VERSION="${TABPFN_VERSION:-6.3.1}"
HF_HUB_SPEC="${HF_HUB_SPEC:-huggingface_hub[cli]>=0.30.0}"
EXTRA_PIP_PACKAGES="${EXTRA_PIP_PACKAGES:-delu tomli-w rtdl_num_embeddings category-encoders rtdl-revisiting-models tensorboard loguru tomli pytest}"
TMP_REQ="$(mktemp)"
trap 'rm -f "$TMP_REQ"' EXIT

if ! command -v "$CONDA_BIN" >/dev/null 2>&1; then
  echo "conda not found on PATH. Set CONDA_BIN or install Miniconda/Anaconda first." >&2
  exit 1
fi

echo "Creating conda prefix env at $VENV_PATH"
"$CONDA_BIN" create -y -p "$VENV_PATH" \
  "python=${PYTHON_VERSION}" \
  "$TORCH_SPEC" \
  "$TORCHVISION_SPEC" \
  "$TORCHAUDIO_SPEC" \
  "$FAISS_SPEC" \
  "$LIBNVJITLINK_SPEC" \
  pip \
  -c pytorch -c nvidia -c rapidsai -c conda-forge --override-channels

PYTHON_BIN="$VENV_PATH/bin/python"

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

"$PYTHON_BIN" - <<'PY' "$ROOT_DIR/requirements.txt" "$TMP_REQ"
from pathlib import Path
import re
import sys

src = Path(sys.argv[1]).read_text().splitlines()
dst = Path(sys.argv[2])
skip = re.compile(r"^(torch|torchvision|torchaudio)([<>=!~].*)?$")
kept = [line for line in src if line.strip() and not skip.match(line.strip())]
dst.write_text("\n".join(kept) + "\n")
print(f"Wrote filtered requirements to {dst}")
PY

"$PYTHON_BIN" -m pip install -r "$TMP_REQ"
"$PYTHON_BIN" -m pip install \
  "tabpfn==${TABPFN_VERSION}" \
  "${HF_HUB_SPEC}" \
  ${EXTRA_PIP_PACKAGES}
"$PYTHON_BIN" -m pip check

"$PYTHON_BIN" - <<'PY'
import importlib

packages = ["torch", "faiss", "tabpfn", "huggingface_hub", "openml"]
for name in packages:
    mod = importlib.import_module(name)
    print(name, getattr(mod, "__version__", "unknown"))

import faiss
import torch

print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
print("faiss_gpu_symbols", hasattr(faiss, "StandardGpuResources"), hasattr(faiss, "index_cpu_to_gpu"))
if torch.cuda.is_available():
    print("cuda_device_name", torch.cuda.get_device_name(0))
PY

echo
echo "Bootstrap complete."
echo "Next steps:"
echo "  source migration/activate_h200_env.sh"
echo "  bash migration/setup_git_and_cli.sh"
echo "  python migration/validate_h200_stack.py --torch-smoke --git-smoke --openml-smoke --openml-download"
