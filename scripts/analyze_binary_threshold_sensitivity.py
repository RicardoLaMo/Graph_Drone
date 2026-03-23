#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, brier_score_loss


def _load_prediction_cache(cache_dir: Path, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    matches = sorted(cache_dir.glob(f"{dataset}__fold*__graphdrone.json"))
    if not matches:
        raise FileNotFoundError(f"No graphdrone cache found for dataset={dataset} in {cache_dir}")
    if len(matches) != 1:
        raise ValueError(f"Expected one cache for dataset={dataset}, found {len(matches)} in {cache_dir}")
    payload = json.loads(matches[0].read_text())
    preds = payload.get("predictions")
    if not preds:
        raise ValueError(
            f"Cache {matches[0]} does not contain predictions. Re-run with GRAPHDRONE_SAVE_CLASSIFICATION_PREDICTIONS=1."
        )
    y_true = np.asarray(preds["y_true"], dtype=int)
    y_pred_proba = np.asarray(preds["y_pred_proba"], dtype=float)
    if y_pred_proba.ndim != 2 or y_pred_proba.shape[1] != 2:
        raise ValueError(f"Dataset {dataset} is not binary or has malformed probabilities: shape={y_pred_proba.shape}")
    return y_true, y_pred_proba[:, 1]


def _sweep_rows(label: str, dataset: str, y_true: np.ndarray, pos_proba: np.ndarray, thresholds: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    for threshold in thresholds:
        y_pred = (pos_proba >= threshold).astype(int)
        rows.append(
            {
                "label": label,
                "dataset": dataset,
                "threshold": float(threshold),
                "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            }
        )
    return rows


def _summary_row(label: str, dataset: str, y_true: np.ndarray, pos_proba: np.ndarray, sweep_df: pd.DataFrame) -> dict:
    label_df = sweep_df[(sweep_df["label"] == label) & (sweep_df["dataset"] == dataset)].copy()
    best = label_df.sort_values(["f1_macro", "threshold"], ascending=[False, True]).iloc[0]
    fixed = label_df[np.isclose(label_df["threshold"], 0.5)].iloc[0]
    return {
        "label": label,
        "dataset": dataset,
        "default_f1_macro": float(fixed["f1_macro"]),
        "best_f1_macro": float(best["f1_macro"]),
        "best_threshold": float(best["threshold"]),
        "f1_gain_vs_default": float(best["f1_macro"] - fixed["f1_macro"]),
        "log_loss": float(log_loss(y_true, np.column_stack([1.0 - pos_proba, pos_proba]))),
        "brier": float(brier_score_loss(y_true, pos_proba)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze binary threshold sensitivity from GraphDrone cache predictions.")
    parser.add_argument("--champion-cache-dir", type=Path, required=True)
    parser.add_argument("--challenger-cache-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-count", type=int, default=19)
    args = parser.parse_args()

    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_count)
    rows: list[dict] = []
    summaries: list[dict] = []

    for dataset in args.datasets:
        champion_y, champion_pos = _load_prediction_cache(args.champion_cache_dir, dataset)
        challenger_y, challenger_pos = _load_prediction_cache(args.challenger_cache_dir, dataset)
        if not np.array_equal(champion_y, challenger_y):
            raise ValueError(f"Champion/challenger labels differ for dataset={dataset}")

        rows.extend(_sweep_rows("champion", dataset, champion_y, champion_pos, thresholds))
        rows.extend(_sweep_rows("challenger", dataset, challenger_y, challenger_pos, thresholds))
        sweep_df = pd.DataFrame(rows)
        summaries.append(_summary_row("champion", dataset, champion_y, champion_pos, sweep_df))
        summaries.append(_summary_row("challenger", dataset, challenger_y, challenger_pos, sweep_df))

    sweep_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summaries)
    merged = summary_df.pivot(index="dataset", columns="label")
    merged.columns = [f"{metric}_{label}" for metric, label in merged.columns]
    merged = merged.reset_index()
    if not merged.empty:
        merged["best_f1_delta"] = merged["best_f1_macro_challenger"] - merged["best_f1_macro_champion"]
        merged["default_f1_delta"] = merged["default_f1_macro_challenger"] - merged["default_f1_macro_champion"]
        merged["best_threshold_shift"] = merged["best_threshold_challenger"] - merged["best_threshold_champion"]
        merged["log_loss_delta"] = merged["log_loss_challenger"] - merged["log_loss_champion"]
        merged["brier_delta"] = merged["brier_challenger"] - merged["brier_champion"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = args.output_dir / "threshold_sweep.csv"
    summary_path = args.output_dir / "threshold_summary.csv"
    report_path = args.output_dir / "threshold_summary.json"
    sweep_df.to_csv(sweep_path, index=False)
    merged.to_csv(summary_path, index=False)
    report = {
        "datasets": args.datasets,
        "threshold_grid": [float(x) for x in thresholds],
        "summary_csv": str(summary_path),
        "sweep_csv": str(sweep_path),
        "per_dataset": merged.to_dict(orient="records"),
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
