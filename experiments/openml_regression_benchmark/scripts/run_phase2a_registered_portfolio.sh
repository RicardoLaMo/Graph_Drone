#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG_PATH="${REPO_ROOT}/experiments/openml_regression_benchmark/configs/phase2a_registered_portfolio.json"
MODE="${1:-full}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  exit 1
fi

mapfile -t DATASETS < <(python - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path("experiments/openml_regression_benchmark/configs/phase2a_registered_portfolio.json").read_text())
for item in cfg["datasets"]:
    print(item)
PY
)

mapfile -t MODELS < <(python - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path("experiments/openml_regression_benchmark/configs/phase2a_registered_portfolio.json").read_text())
for item in cfg["models"]:
    print(item)
PY
)

if [[ "${MODE}" == "smoke" ]]; then
  SMOKE_FLAG="--smoke"
  FOLDS=(0)
  OUTPUT_ROOT="${REPO_ROOT}/experiments/openml_regression_benchmark/reports_phase2a_smoke_portfolio"
else
  SMOKE_FLAG=""
  FOLDS=(0 1 2)
  OUTPUT_ROOT="${REPO_ROOT}/experiments/openml_regression_benchmark/reports_phase2a_full_portfolio"
fi

/home/wliu23/projects/Graph_Drone/.venv-h200/bin/python \
  "${REPO_ROOT}/experiments/openml_regression_benchmark/scripts/run_openml_suite.py" \
  --datasets "${DATASETS[@]}" \
  --folds "${FOLDS[@]}" \
  --models "${MODELS[@]}" \
  --gpus auto \
  --max-concurrent-jobs 8 \
  --output-root "${OUTPUT_ROOT}" \
  ${SMOKE_FLAG}
