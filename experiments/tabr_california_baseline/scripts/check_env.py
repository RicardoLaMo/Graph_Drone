from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import torch


def main():
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/tabr_california_baseline/artifacts/environment.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import faiss

        payload["faiss"] = faiss.__version__
    except Exception as err:
        payload["faiss"] = f"missing: {type(err).__name__}: {err}"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()

