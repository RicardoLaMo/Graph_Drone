from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from experiments.california_geo_segmentation.src.segmentation import SegmentationResult


@dataclass(frozen=True)
class EvalResult:
    model: str
    rmse: float
    mae: float
    r2: float
    n_features: int
    note: str


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def _fit_hgbr(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    model = HistGradientBoostingRegressor(
        random_state=42,
        learning_rate=0.05,
        max_depth=6,
        max_iter=500,
        l2_regularization=1e-3,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
    )
    model.fit(X[train_idx], y[train_idx])
    return model.predict(X[test_idx]).astype(np.float32)


def evaluate_models(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    segmentations: Dict[str, SegmentationResult],
    shuffled_segmentations: Dict[str, SegmentationResult] | None = None,
) -> list[EvalResult]:
    results: list[EvalResult] = []

    pred_raw = _fit_hgbr(X, y, train_idx, test_idx)
    rmse, mae, r2 = metrics(y[test_idx], pred_raw)
    results.append(EvalResult("B1_HGBR_raw", rmse, mae, r2, X.shape[1], "raw scaled features"))

    for name, seg in segmentations.items():
        seg_mean = seg.features[:, 0]
        rmse, mae, r2 = metrics(y[test_idx], seg_mean[test_idx])
        results.append(EvalResult(f"S0_{name}_mean", rmse, mae, r2, 1, "segment train mean only"))

        X_aug = np.concatenate([X, seg.features], axis=1)
        pred = _fit_hgbr(X_aug, y, train_idx, test_idx)
        rmse, mae, r2 = metrics(y[test_idx], pred)
        results.append(EvalResult(f"H1_{name}_aug", rmse, mae, r2, X_aug.shape[1], "raw + segment priors"))

    all_prior_feats = np.concatenate([seg.features for seg in segmentations.values()], axis=1)
    X_all = np.concatenate([X, all_prior_feats], axis=1)
    pred_all = _fit_hgbr(X_all, y, train_idx, test_idx)
    rmse, mae, r2 = metrics(y[test_idx], pred_all)
    results.append(EvalResult("H2_all_geo_priors", rmse, mae, r2, X_all.shape[1], "raw + all geo priors"))

    target_only_feats = np.concatenate([seg.features[:, :2] for seg in segmentations.values()], axis=1)
    X_target_only = np.concatenate([X, target_only_feats], axis=1)
    pred_target_only = _fit_hgbr(X_target_only, y, train_idx, test_idx)
    rmse, mae, r2 = metrics(y[test_idx], pred_target_only)
    results.append(
        EvalResult(
            "H4_all_geo_target_stats_only",
            rmse,
            mae,
            r2,
            X_target_only.shape[1],
            "raw + per-segment target mean/std only",
        )
    )

    structure_only_parts = []
    for seg in segmentations.values():
        if seg.features.shape[1] > 2:
            structure_only_parts.append(seg.features[:, 2:])
    if structure_only_parts:
        structure_only_feats = np.concatenate(structure_only_parts, axis=1)
        X_structure_only = np.concatenate([X, structure_only_feats], axis=1)
        pred_structure_only = _fit_hgbr(X_structure_only, y, train_idx, test_idx)
        rmse, mae, r2 = metrics(y[test_idx], pred_structure_only)
        results.append(
            EvalResult(
                "H5_all_geo_structure_only",
                rmse,
                mae,
                r2,
                X_structure_only.shape[1],
                "raw + segment count/density/centroid structure only",
            )
        )

    if shuffled_segmentations is not None:
        all_shuffled_feats = np.concatenate(
            [seg.features for seg in shuffled_segmentations.values()],
            axis=1,
        )
        X_shuffled = np.concatenate([X, all_shuffled_feats], axis=1)
        pred_shuffled = _fit_hgbr(X_shuffled, y, train_idx, test_idx)
        rmse, mae, r2 = metrics(y[test_idx], pred_shuffled)
        results.append(
            EvalResult(
                "H3_all_geo_priors_shuffled",
                rmse,
                mae,
                r2,
                X_shuffled.shape[1],
                "raw + geo priors built from shuffled train targets",
            )
        )

    return results


def results_to_frame(results: list[EvalResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": r.model,
                "rmse": r.rmse,
                "mae": r.mae,
                "r2": r.r2,
                "n_features": r.n_features,
                "note": r.note,
            }
            for r in results
        ]
    ).sort_values("rmse", ascending=True, ignore_index=True)
