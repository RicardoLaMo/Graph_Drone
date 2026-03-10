from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from experiments.openml_regression_benchmark.src.openml_tasks import PreparedOpenMLSplit

from .tabpfn_baseline import select_train_subset


def _score_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _build_feature_frame(
    *,
    X_num: np.ndarray | None,
    X_cat: np.ndarray | None,
    num_feature_names: tuple[str, ...],
    cat_feature_names: tuple[str, ...],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    row_count: int | None = None
    if X_num is not None:
        row_count = len(X_num)
        frames.append(pd.DataFrame(X_num, columns=list(num_feature_names)).astype(np.float32))
    if X_cat is not None:
        if row_count is not None and len(X_cat) != row_count:
            raise ValueError("Numeric and categorical feature blocks must have the same row count")
        row_count = len(X_cat)
        cat_df = pd.DataFrame(X_cat, columns=list(cat_feature_names)).astype(str)
        for column in cat_df.columns:
            cat_df[column] = pd.Categorical(cat_df[column])
        frames.append(cat_df)
    if not frames:
        return pd.DataFrame(index=pd.RangeIndex(row_count or 0))
    feature_frame = pd.concat(frames, axis=1)
    if row_count is not None and len(feature_frame) != row_count:
        raise ValueError("Built AutoGluon feature frame has an unexpected row count")
    return feature_frame


def build_autogluon_frames(
    split: PreparedOpenMLSplit,
    *,
    train_subset: np.ndarray | None = None,
    label_name: str = "__target__",
) -> dict[str, pd.DataFrame]:
    train_rows = slice(None) if train_subset is None else train_subset
    train_df = _build_feature_frame(
        X_num=None if split.X_num_train is None else split.X_num_train[train_rows],
        X_cat=None if split.X_cat_train is None else split.X_cat_train[train_rows],
        num_feature_names=split.num_feature_names,
        cat_feature_names=split.cat_feature_names,
    )
    val_df = _build_feature_frame(
        X_num=split.X_num_val,
        X_cat=split.X_cat_val,
        num_feature_names=split.num_feature_names,
        cat_feature_names=split.cat_feature_names,
    )
    test_df = _build_feature_frame(
        X_num=split.X_num_test,
        X_cat=split.X_cat_test,
        num_feature_names=split.num_feature_names,
        cat_feature_names=split.cat_feature_names,
    )
    train_df[label_name] = split.y_train if train_subset is None else split.y_train[train_subset]
    val_df[label_name] = split.y_val
    test_df[label_name] = split.y_test
    return {"train": train_df, "val": val_df, "test": test_df}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _leaderboard_preview(df: pd.DataFrame, *, limit: int = 10) -> list[dict[str, Any]]:
    keep_columns = [column for column in ("model", "score_test", "score_val", "fit_time", "pred_time_test", "stack_level", "can_infer") if column in df.columns]
    preview: list[dict[str, Any]] = []
    for _, row in df.loc[:, keep_columns].head(limit).iterrows():
        preview.append({column: _json_safe(value) for column, value in row.to_dict().items()})
    return preview


@dataclass(frozen=True)
class AutoGluonPredictionResult:
    pred_val: np.ndarray
    pred_test: np.ndarray
    metrics: dict[str, Any]


def predict_autogluon_regression(
    split: PreparedOpenMLSplit,
    *,
    seed: int = 42,
    max_train_samples: int | None = None,
    time_limit: float | None = 600.0,
    presets: str = "medium_quality",
    num_cpus: int | str = 4,
    num_gpus: int | str = "auto",
    fit_strategy: str = "sequential",
    artifacts_path: Path | None = None,
    verbosity: int = 1,
) -> AutoGluonPredictionResult:
    from autogluon.tabular import TabularPredictor

    label_name = "__target__"
    train_subset = select_train_subset(
        n_train=len(split.X_train),
        max_train_samples=max_train_samples,
        seed=seed,
    )
    frames = build_autogluon_frames(split, train_subset=train_subset, label_name=label_name)

    predictor = TabularPredictor(
        label=label_name,
        problem_type="regression",
        eval_metric="root_mean_squared_error",
        path=None if artifacts_path is None else str(artifacts_path),
        verbosity=verbosity,
    )
    started_at = time.time()
    predictor.fit(
        train_data=frames["train"],
        tuning_data=frames["val"],
        time_limit=time_limit,
        presets=presets,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        fit_strategy=fit_strategy,
        ag_args_ensemble={"model_random_seed": seed, "vary_seed_across_folds": False},
    )
    duration_seconds = round(time.time() - started_at, 3)

    val_features = frames["val"].drop(columns=[label_name])
    test_features = frames["test"].drop(columns=[label_name])
    pred_val = predictor.predict(val_features).to_numpy(dtype=np.float32)
    pred_test = predictor.predict(test_features).to_numpy(dtype=np.float32)
    leaderboard_test = predictor.leaderboard(frames["test"], silent=True)

    metrics = {
        "seed": seed,
        "train_samples_used": int(len(frames["train"])),
        "time_limit": None if time_limit is None else float(time_limit),
        "presets": presets,
        "num_cpus": num_cpus,
        "num_gpus": num_gpus,
        "fit_strategy": fit_strategy,
        "ensemble_seed": seed,
        "duration_seconds": duration_seconds,
        "best_model": predictor.model_best,
        "model_names": predictor.model_names(),
        "leaderboard_test_head": _leaderboard_preview(leaderboard_test),
        "val": _score_regression(split.y_val, pred_val),
        "test": _score_regression(split.y_test, pred_test),
    }
    return AutoGluonPredictionResult(
        pred_val=pred_val,
        pred_test=pred_test,
        metrics=metrics,
    )
