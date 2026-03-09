from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .aligned_california import SplitData


def select_train_subset(
    n_train: int,
    max_train_samples: int | None,
    seed: int,
) -> np.ndarray:
    if max_train_samples is None or max_train_samples >= n_train:
        return np.arange(n_train, dtype=np.int64)
    rng = np.random.RandomState(seed)
    idx = rng.choice(n_train, size=max_train_samples, replace=False)
    idx.sort()
    return idx.astype(np.int64)


def _score_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


@dataclass(frozen=True)
class TabPFNRunConfig:
    seed: int = 42
    max_train_samples: int | None = 8000
    n_estimators: int = 8
    max_eval_rows: int | None = None
    device: str = "cpu"
    n_preprocessing_jobs: int = 1


def evaluate_tabpfn_regression(
    split: SplitData,
    seed: int = 42,
    max_train_samples: int | None = 8000,
    n_estimators: int = 8,
    max_eval_rows: int | None = None,
    device: str = "cpu",
    n_preprocessing_jobs: int = 1,
) -> dict:
    from tabpfn import TabPFNRegressor

    train_subset = select_train_subset(
        n_train=len(split.X_train),
        max_train_samples=max_train_samples,
        seed=seed,
    )
    X_train = split.X_train[train_subset]
    y_train = split.y_train[train_subset]

    X_val = split.X_val
    y_val = split.y_val
    X_test = split.X_test
    y_test = split.y_test

    if max_eval_rows is not None:
        X_val = X_val[:max_eval_rows]
        y_val = y_val[:max_eval_rows]
        X_test = X_test[:max_eval_rows]
        y_test = y_test[:max_eval_rows]

    if device == "cpu" and len(X_train) > 1000:
        os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

    model = TabPFNRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        device=device,
        n_preprocessing_jobs=n_preprocessing_jobs,
        ignore_pretraining_limits=len(X_train) > 1000,
    )
    model.fit(X_train, y_train)
    pred_val = model.predict(X_val).astype(np.float32)
    pred_test = model.predict(X_test).astype(np.float32)

    return {
        "seed": seed,
        "train_samples_used": int(len(X_train)),
        "n_estimators": int(n_estimators),
        "device": device,
        "max_eval_rows": None if max_eval_rows is None else int(max_eval_rows),
        "val": _score_regression(y_val, pred_val),
        "test": _score_regression(y_test, pred_test),
    }


def write_tabpfn_report(
    output_path: Path,
    metrics: dict,
    *,
    tabr_rmse: float,
    tabm_rmse: float,
    a6f_rmse: float,
) -> None:
    lines = [
        "# TabPFN Aligned California Report",
        "",
        "## Result",
        "",
        f"- TabPFN_on_our_split: test RMSE `{metrics['test']['rmse']:.4f}`",
        f"- val RMSE `{metrics['val']['rmse']:.4f}`",
        f"- train samples used `{metrics['train_samples_used']}`",
        f"- n_estimators `{metrics['n_estimators']}`",
        "",
        "## Comparison",
        "",
        f"- vs TabR_on_our_split `{tabr_rmse:.4f}`: delta `{metrics['test']['rmse'] - tabr_rmse:+.4f}`",
        f"- vs TabM_on_our_split `{tabm_rmse:.4f}`: delta `{metrics['test']['rmse'] - tabm_rmse:+.4f}`",
        f"- vs A6f `{a6f_rmse:.4f}`: delta `{metrics['test']['rmse'] - a6f_rmse:+.4f}`",
        "",
        "## Notes",
        "",
        "- This run uses the aligned California split and feature edits from the foundation comparison branch.",
        "- CPU-only TabPFN runs above 1000 rows require the current package override; this run records the exact train cap used.",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def write_tabpfn_json(output_path: Path, metrics: dict) -> None:
    output_path.write_text(json.dumps(metrics, indent=2) + "\n")
