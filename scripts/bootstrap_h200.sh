#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${1:-$ROOT_DIR/.venv-h200}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
TORCH_CHANNEL="${TORCH_CHANNEL:-cu124}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.6.0}"
TMP_REQ="$(mktemp)"
trap 'rm -f "$TMP_REQ"' EXIT

echo "Creating virtualenv at $VENV_PATH"
"$PYTHON_BIN" -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

python -m pip install --upgrade pip setuptools wheel

python - <<'PY' "$ROOT_DIR/requirements.txt" "$TMP_REQ"
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

python -m pip install -r "$TMP_REQ"
python -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL}"
python -m pip install tabpfn

python - <<'PY'
import torch
print("torch_version", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_name", torch.cuda.get_device_name(0))
PY

echo "H200 bootstrap complete. Activate with:"
echo "source \"$VENV_PATH/bin/activate\""
