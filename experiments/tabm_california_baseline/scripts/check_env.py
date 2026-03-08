from __future__ import annotations

import json
import platform
from pathlib import Path

import torch


def main() -> None:
    output_path = Path("experiments/tabm_california_baseline/artifacts/environment.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
