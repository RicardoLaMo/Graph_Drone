from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.openml_classification_benchmark.src.foundation_config import (
    prepare_foundation_config,
    resolve_tabr_config_name,
)
from experiments.openml_classification_benchmark.src.openml_tasks import (
    available_dataset_keys,
    build_openml_classification_split,
    dataset_run_tag,
    split_summary,
    write_tabr_foundation_dataset,
)
from experiments.openml_classification_benchmark.src.prediction_metrics import (
    classification_metrics_from_raw_prediction,
)
from experiments.tab_foundation_compare.scripts.run_tabr_aligned import _patch_tabr_for_modern_env
from experiments.tab_foundation_compare.src.runtime_support import (
    default_foundation_python,
    default_single_gpu_cuda_visible_devices,
    default_tabr_upstream_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TabR on OpenML classification task splits")
    parser.add_argument("--dataset", choices=available_dataset_keys(), required=True)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--upstream-root", type=Path, default=default_tabr_upstream_root(REPO_ROOT))
    parser.add_argument("--venv-python", type=Path, default=default_foundation_python(REPO_ROOT))
    parser.add_argument("--config", default="auto")
    parser.add_argument("--device-policy", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "openml_classification_benchmark" / "reports",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = build_openml_classification_split(
        args.dataset,
        repeat=args.repeat,
        fold=args.fold,
        split_seed=args.split_seed,
        smoke=args.smoke,
    )
    run_name = dataset_run_tag(args.dataset, repeat=args.repeat, fold=args.fold, smoke=args.smoke)
    output_dir = (args.output_root / run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    local_data_dir = output_dir / f"tabr_dataset__{run_name}"
    write_tabr_foundation_dataset(local_data_dir, split)

    resolved_config = resolve_tabr_config_name(split) if args.config == "auto" else args.config
    source_config = args.upstream_root / "exp" / "tabr" / f"{resolved_config}.toml"
    if not source_config.exists():
        raise FileNotFoundError(source_config)
    local_config = output_dir / "tabr_config.toml"
    tabr_cat_policy = None
    if split.X_cat_train is not None:
        tabr_cat_policy = "one-hot" if split.X_num_train is None else "ordinal"
    prepare_foundation_config(
        source_config=source_config,
        output_config=local_config,
        data_path=str(local_data_dir),
        seed=args.seed,
        smoke=args.smoke,
        num_policy="quantile" if split.X_num_train is not None else None,
        cat_policy=tabr_cat_policy,
        null_toml_token="__null__" if split.X_cat_train is None or split.X_num_train is None else None,
    )

    _patch_tabr_for_modern_env(args.upstream_root)

    env = os.environ.copy()
    env["PROJECT_DIR"] = str(args.upstream_root)
    if args.device_policy == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", default_single_gpu_cuda_visible_devices())
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("TQDM_DISABLE", "1")

    stdout_path = output_dir / "tabr.stdout.log"
    stderr_path = output_dir / "tabr.stderr.log"
    started_at = time.time()
    command = [str(args.venv_python), "bin/tabr.py", str(local_config), "--force"]
    with stdout_path.open("w") as stdout_handle, stderr_path.open("w") as stderr_handle:
        result = subprocess.run(
            command,
            cwd=args.upstream_root,
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    run_output_dir = local_config.with_suffix("")
    upstream_report = json.loads((run_output_dir / "report.json").read_text())
    predictions = dict(np.load(run_output_dir / "predictions.npz"))
    prediction_type = str(upstream_report.get("prediction_type", "logits"))
    val_metrics = classification_metrics_from_raw_prediction(
        split.y_val,
        predictions["val"],
        prediction_type=prediction_type,
        class_labels=tuple(str(v) for v in split.class_names),
    )
    test_metrics = classification_metrics_from_raw_prediction(
        split.y_test,
        predictions["test"],
        prediction_type=prediction_type,
        class_labels=tuple(str(v) for v in split.class_names),
    )
    payload = {
        "run_name": run_name,
        "model": "TabR",
        "seed": args.seed,
        "dataset": split_summary(split),
        "config": resolved_config,
        "device_policy": args.device_policy,
        "duration_seconds": round(time.time() - started_at, 3),
        "rows": [
            {
                "model": "TabR",
                "val_accuracy": val_metrics["accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "val_macro_f1": val_metrics["f1_macro"],
                "test_macro_f1": test_metrics["f1_macro"],
                "val_roc_auc": val_metrics["roc_auc"],
                "test_roc_auc": test_metrics["roc_auc"],
                "val_pr_auc": val_metrics["pr_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "val_log_loss": val_metrics["log_loss"],
                "test_log_loss": test_metrics["log_loss"],
                "notes": "metrics recomputed from upstream TabR predictions.npz",
            }
        ],
        "upstream_metrics": upstream_report["metrics"],
        "best_epoch": upstream_report.get("best_epoch"),
        "upstream_report_path": str(run_output_dir / "report.json"),
        "prediction_type": prediction_type,
    }
    (output_dir / "tabr_results.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
