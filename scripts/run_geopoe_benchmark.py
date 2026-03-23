#!/usr/bin/env python3
"""
Smart Benchmark Runner — addresses GitHub Issue #16
=====================================================
Problems solved:
  1. Caching  — each (dataset, fold, method) result is saved immediately after completion.
     On crash/restart, completed tasks are skipped; no more "groundhog day" re-runs.
  2. Granularity — per-dataset metrics: RMSE + R² for regression,
     F1-macro + AUC-ROC + PR-AUC for classification.
  3. Win-rate table — per-dataset head-to-head vs each baseline, not just a single ELO number.

Usage:
    # Quick smoke-test (3 datasets, 1 fold)
    PYTHONPATH=src python scripts/run_smart_benchmark.py --quick

    # Full regression suite (3 seeds)
    PYTHONPATH=src python scripts/run_smart_benchmark.py \\
        --tasks regression --folds 0 1 2 --cache-dir eval/smart_cache

    # Resume after a crash — already-cached tasks are skipped automatically
    PYTHONPATH=src python scripts/run_smart_benchmark.py --tasks regression --folds 0 1 2
"""

import sys
import json
import hashlib
import argparse
import os
import traceback
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import openml
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    f1_score, roc_auc_score, average_precision_score,
)
from sklearn.preprocessing import OrdinalEncoder, label_binarize

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from graphdrone_fit.model import GraphDrone
from graphdrone_fit.config import (
    GraphDroneConfig,
)
from graphdrone_fit.expert_factory import ExpertBuildSpec, IdentitySelectorAdapter
from graphdrone_fit.presets import build_graphdrone_config_from_preset
from graphdrone_fit.view_descriptor import ViewDescriptor

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

REGRESSION_DATASETS = {
    "california":    {"openml_id": 541,   "type": "regression"},
    "diamonds":      {"openml_id": 42225, "type": "regression"},
    "house_prices":  {"openml_id": 42165, "type": "regression"},
    "elevators":     {"openml_id": 216,   "type": "regression"},
    "cpu_act":       {"openml_id": 761,   "type": "regression"},  # ~8k rows, continuous CPU activity
    "kin8nm":        {"openml_id": 189,   "type": "regression"},  # ~8k rows, kinematic regression
}

CLASSIFICATION_DATASETS = {
    "segment":       {"openml_id": 40984, "type": "classification"},
    "mfeat_factors": {"openml_id": 12,    "type": "classification"},
    "pendigits":     {"openml_id": 32,    "type": "classification"},
    "optdigits":     {"openml_id": 28,    "type": "classification"},
    "diabetes":      {"openml_id": 37,    "type": "classification"},
    "credit_g":      {"openml_id": 31,    "type": "classification"},
}

ALL_DATASETS = {**REGRESSION_DATASETS, **CLASSIFICATION_DATASETS}

QUICK_DATASETS = {
    "california":    REGRESSION_DATASETS["california"],
    "cpu_act":       REGRESSION_DATASETS["cpu_act"],
    "pendigits":     CLASSIFICATION_DATASETS["pendigits"],
}

GRAPHDRONE_VERSION = os.getenv("GRAPHDRONE_VERSION_OVERRIDE", "v1-geopoe-2026.03.22-afc-d-regonly-v1")
GRAPHDRONE_PRESET = os.getenv("GRAPHDRONE_PRESET", "afc_candidate")


def _graphdrone_config(*, n_classes: int = 1, default_router_kind: str, dataset_key: str | None = None) -> GraphDroneConfig:
    return build_graphdrone_config_from_preset(
        preset=GRAPHDRONE_PRESET,
        n_classes=n_classes,
        default_router_kind=default_router_kind,
        task_prior_dataset_key=dataset_key,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _preprocess(X: pd.DataFrame, y: pd.Series):
    X = X.copy()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if cat_cols:
        X[cat_cols] = X[cat_cols].astype(str)
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = enc.fit_transform(X[cat_cols])
    X = X.apply(lambda c: c.fillna(c.median()) if c.notna().any() else c.fillna(0))
    X_arr = X.values.astype(np.float32)

    if pd.api.types.is_categorical_dtype(y) or y.dtype == object:
        from sklearn.preprocessing import LabelEncoder
        y_arr = LabelEncoder().fit_transform(y.astype(str))
    else:
        y_arr = y.fillna(y.mean()).values
    return X_arr, y_arr


def load_dataset(name: str, max_samples: int = 5000):
    meta = ALL_DATASETS[name]
    ds = openml.datasets.get_dataset(meta["openml_id"])
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
    X_arr, y_arr = _preprocess(X, y)
    if len(X_arr) > max_samples:
        _, X_arr, _, y_arr = train_test_split(
            X_arr, y_arr, test_size=max_samples, random_state=0,
        )
    return X_arr, y_arr, meta["type"]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(dataset: str, fold: int, method: str) -> str:
    raw = f"{dataset}|{fold}|{method}|{GRAPHDRONE_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_path(cache_dir: Path, dataset: str, fold: int, method: str) -> Path:
    safe = method.replace(" ", "_")
    return cache_dir / f"{dataset}__fold{fold}__{safe}.json"


def load_cache(path: Path) -> Optional[dict]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def save_cache(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def regression_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, y_pred_proba, y_pred_labels) -> dict:
    n_classes = y_pred_proba.shape[1]
    avg = "macro"
    f1 = float(f1_score(y_true, y_pred_labels, average=avg, zero_division=0))
    try:
        if n_classes == 2:
            auc = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
            pr  = float(average_precision_score(y_true, y_pred_proba[:, 1]))
        else:
            y_bin = label_binarize(y_true, classes=list(range(n_classes)))
            auc = float(roc_auc_score(y_bin, y_pred_proba, multi_class="ovr", average=avg))
            pr  = float(average_precision_score(y_bin, y_pred_proba, average=avg))
    except Exception:
        auc = float("nan")
        pr  = float("nan")
    return {"f1_macro": f1, "auc_roc": auc, "pr_auc": pr}


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_tabpfn(X_tr, y_tr, X_te, task_type: str):
    """TabPFN v2.5 with its default configuration (n_estimators=8 out of the box).
    Only override: ignore_pretraining_limits=True, needed because our datasets exceed
    the 1000-row safety limit built into TabPFN."""
    if task_type == "regression":
        from tabpfn import TabPFNRegressor
        m = TabPFNRegressor(ignore_pretraining_limits=len(X_tr) > 1000)
        m.fit(X_tr, y_tr)
        return m.predict(X_te), None, {}
    else:
        from tabpfn import TabPFNClassifier
        m = TabPFNClassifier(ignore_pretraining_limits=len(X_tr) > 1000)
        m.fit(X_tr, y_tr.astype(int))
        proba = m.predict_proba(X_te)
        return proba, np.argmax(proba, axis=1), {}


def _diagnostic_payload(diagnostics: dict[str, object]) -> dict[str, object]:
    keep_keys = (
        "router_kind",
        "n_experts",
        "n_specialists",
        "mean_defer_prob",
        "effective_defer_rate",
        "early_exit",
        "exit_frac",
        "legitimacy_metric",
        "legitimacy_threshold",
        "legitimacy_score_mean",
        "router_skipped",
        "router_nonfinite_fallback",
        "regression_router_fallback_stage",
        "regression_router_fallback_reason",
        "validation_router_tokens_finite_flag",
        "validation_router_predictions_finite_flag",
        "validation_router_targets_finite_flag",
        "validation_anchor_mse_finite_flag",
        "prediction_router_tokens_finite_flag",
        "prediction_router_weights_finite_flag",
        "prediction_router_defer_finite_flag",
        "mean_ot_cost",
        "mean_specialist_validity",
        "closed_specialist_frac",
        "mean_specialist_mass",
        "mean_anchor_attention_weight",
        "alignment_aux_loss",
        "alignment_cosine_pre",
        "alignment_cosine_post",
        "alignment_cosine_gain",
        "task_prior_enabled",
        "task_prior_strength",
        "task_prior_norm",
        "task_prior_training_objective",
        "task_prior_encoder_kind",
        "task_prior_query_dataset",
        "task_prior_top_neighbor",
        "task_prior_top_neighbor_prob",
        "task_prior_base_top_neighbor",
        "task_prior_base_top_neighbor_prob",
        "task_prior_entropy",
        "task_prior_exact_reuse_available",
        "task_prior_exact_reuse_blend",
        "task_prior_exact_reuse_used",
        "task_prior_feedback_used",
        "task_prior_feedback_top_source",
        "task_prior_feedback_top_source_weight",
        "validation_best_specialist_advantage_score",
        "validation_weighted_specialist_advantage_score",
        "validation_defer_weighted_specialist_advantage_score",
        "validation_top_specialist_advantage_score",
        "validation_positive_specialist_opportunity_score",
        "validation_residual_usefulness_gap",
        "validation_positive_specialist_mass",
        "validation_top_specialist_positive_rate",
        "validation_residual_usefulness_lambda",
    )
    payload: dict[str, object] = {}
    for key in keep_keys:
        if key not in diagnostics:
            continue
        value = diagnostics[key]
        if isinstance(value, (np.floating, float)):
            payload[key] = float(value)
        elif isinstance(value, (np.integer, int)):
            payload[key] = int(value)
        elif isinstance(value, (np.bool_, bool)):
            payload[key] = bool(value)
        elif value is not None:
            payload[key] = str(value)
    for key, value in diagnostics.items():
        if key in payload:
            continue
        if not (
            key.startswith("mean_attention_")
            or key == "non_anchor_attention_entropy"
            or key.startswith("validation_")
        ):
            continue
        if isinstance(value, (np.floating, float)):
            payload[key] = float(value)
        elif isinstance(value, (np.integer, int)):
            payload[key] = int(value)
    return payload


def run_graphdrone(X_tr, y_tr, X_te, task_type: str, *, dataset: str, seed: int = 42, n_classes: int = None):
    """
    Two independent engines:
    - Regression: FULL + 3×SUB TabPFN views, GORA observers, contextual_transformer
      router trained with MSE loss on 10% OOF validation. No tree models.
    - Classification: FULL + 3×SUB TabPFN views, static anchor-boosted GeoPOE
      (anchor_weight=3.0). No learned router — avoids OOF overfitting.
    """
    n = X_tr.shape[1]
    full_idx = tuple(range(n))
    dev = _device()
    params_fp = {"n_estimators": 8, "device": dev}

    rng = np.random.RandomState(seed)
    sub_size = max(1, int(n * 0.7))
    sub_idx = tuple(sorted(rng.choice(n, sub_size, replace=False).tolist()))

    if task_type == "regression":
        # Multi-view regression: FULL + 3×SUB TabPFN, GORA observers, learned router.
        # No tree models — they are weaker than TabPFN and cause the router to mis-route.
        # contextual_transformer router trains on 10% OOF validation with MSE loss;
        # GORA (kappa + LID) is computed in model.py fit() and included in tokens.
        full_spec = ExpertBuildSpec(
            descriptor=ViewDescriptor(
                expert_id="FULL", family="FULL", view_name="Foundation Full",
                is_anchor=True, input_dim=n, input_indices=full_idx,
            ),
            model_kind="foundation_regressor",
            input_adapter=IdentitySelectorAdapter(indices=full_idx),
            model_params=params_fp,
        )
        sub_specs = []
        for sub_seed, sub_frac in [(0, 0.7), (1, 0.7), (2, 0.8)]:
            rng_i = np.random.RandomState(sub_seed)
            sz_i = max(1, int(n * sub_frac))
            idx_i = tuple(sorted(rng_i.choice(n, sz_i, replace=False).tolist()))
            sub_specs.append(ExpertBuildSpec(
                descriptor=ViewDescriptor(
                    expert_id=f"SUB{sub_seed}", family="structural_subspace",
                    view_name=f"Foundation Sub {sub_seed}",
                    input_dim=sz_i, input_indices=idx_i,
                ),
                model_kind="foundation_regressor",
                input_adapter=IdentitySelectorAdapter(indices=idx_i),
                model_params=params_fp,
            ))
        specs = (full_spec, *sub_specs)
        cfg = _graphdrone_config(default_router_kind="contextual_transformer", dataset_key=dataset)
    else:
        # GeoPOE classification: multi-view SUB portfolio + static anchor-boosted GeoPOE
        # 3 SUB views with different seeds/subspace sizes → richer ensemble diversity
        # Static router avoids overfitting on tiny OOF splits (~78-100 samples)
        if n_classes is None:
            n_classes = int(len(np.unique(y_tr)))
        full_spec = ExpertBuildSpec(
            descriptor=ViewDescriptor(
                expert_id="FULL", family="FULL", view_name="Foundation Full",
                is_anchor=True, input_dim=n, input_indices=full_idx,
            ),
            model_kind="foundation_classifier",
            input_adapter=IdentitySelectorAdapter(indices=full_idx),
            model_params=params_fp,
        )
        sub_specs = []
        for sub_seed, sub_frac in [(0, 0.7), (1, 0.7), (2, 0.8)]:
            rng_i = np.random.RandomState(sub_seed)
            sz_i = max(1, int(n * sub_frac))
            idx_i = tuple(sorted(rng_i.choice(n, sz_i, replace=False).tolist()))
            sub_specs.append(ExpertBuildSpec(
                descriptor=ViewDescriptor(
                    expert_id=f"SUB{sub_seed}", family="structural_subspace",
                    view_name=f"Foundation Sub {sub_seed}",
                    input_dim=sz_i, input_indices=idx_i,
                ),
                model_kind="foundation_classifier",
                input_adapter=IdentitySelectorAdapter(indices=idx_i),
                model_params=params_fp,
            ))
        specs = (full_spec, *sub_specs)
        cfg = _graphdrone_config(n_classes=n_classes, default_router_kind="bootstrap_full_only", dataset_key=dataset)

    gd = GraphDrone(cfg)
    # Pass problem_type explicitly so integer-valued regression targets (e.g. cpu_act)
    # are not auto-detected as classification.
    gd.fit(X_tr, y_tr, expert_specs=specs,
           problem_type="regression" if task_type == "regression" else None)
    result = gd.predict(X_te, return_diagnostics=True)
    preds = result.predictions
    diagnostics = _diagnostic_payload(result.diagnostics)

    if task_type == "regression":
        return preds, None, diagnostics
    else:
        proba = preds  # always [N, C] from GeoPOE
        return proba, np.argmax(proba, axis=1), diagnostics


# ---------------------------------------------------------------------------
# Single task runner (one dataset × one fold × all methods)
# ---------------------------------------------------------------------------

DEFAULT_METHODS = ["tabpfn", "graphdrone"]


def run_task(dataset: str, fold: int, cache_dir: Path, max_samples: int, methods: list[str]) -> list[dict]:
    print(f"\n  [{dataset}  fold={fold}]")
    rows = []

    # Load data once per task
    try:
        X, y, task_type = load_dataset(dataset, max_samples=max_samples)
    except Exception as e:
        print(f"    LOAD ERROR: {e}")
        return []

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=fold * 17 + 42)
    # Compute n_classes from full y (before split) so GraphDrone never gets a truncated class range
    global_n_classes = int(len(np.unique(y))) if task_type == "classification" else None

    for method in methods:
        cpath = _cache_path(cache_dir, dataset, fold, method)
        cached = load_cache(cpath)
        key = _cache_key(dataset, fold, method)

        if cached and cached.get("cache_key") == key and cached.get("status") == "ok":
            diagnostics = cached.get("diagnostics", {})
            print(f"    [{method}] CACHED — skipping")
            rows.append({
                **cached["metrics"],
                **diagnostics,
                "dataset": dataset,
                "fold": fold,
                "method": method,
                "task_type": task_type,
                "defer": cached.get("defer", diagnostics.get("mean_defer_prob", float("nan"))),
                "elapsed": cached.get("elapsed", float("nan")),
                "status": "cached",
            })
            continue

        print(f"    [{method}] running...", end=" ", flush=True)
        t0 = time.time()
        try:
            if method == "tabpfn":
                out = run_tabpfn(X_tr, y_tr, X_te, task_type)
            else:
                out = run_graphdrone(X_tr, y_tr, X_te, task_type, dataset=dataset, seed=42, n_classes=global_n_classes)

            elapsed = time.time() - t0

            if task_type == "regression":
                preds, _, diagnostics = out
                metrics = regression_metrics(y_te, preds)
            else:
                proba, labels, diagnostics = out
                metrics = classification_metrics(y_te, proba, labels)
            defer = diagnostics.get("mean_defer_prob", float("nan"))

            print(f"OK ({elapsed:.1f}s)  " +
                  "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()) +
                  (f"  defer={defer:.4f}" if not np.isnan(float(defer)) else ""))

            payload = {
                "cache_key": key,
                "status": "ok",
                "metrics": metrics,
                "diagnostics": diagnostics,
                "defer": float(defer) if not np.isnan(float(defer)) else None,
                "elapsed": elapsed,
            }
            save_cache(cpath, payload)
            rows.append({
                **metrics,
                **diagnostics,
                "dataset": dataset,
                "fold": fold,
                "method": method,
                "task_type": task_type,
                "defer": float(defer),
                "elapsed": elapsed,
                "status": "ok",
            })

        except Exception as e:
            elapsed = time.time() - t0
            msg = str(e)[:200]
            print(f"FAIL ({elapsed:.1f}s): {msg}")
            payload = {"cache_key": key, "status": "fail", "error": msg, "elapsed": elapsed}
            save_cache(cpath, payload)
            rows.append({"dataset": dataset, "fold": fold, "method": method,
                         "task_type": task_type, "status": "fail", "elapsed": elapsed})

    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def compute_elo(df: pd.DataFrame, metric_map: dict[str, str],
                K: int = 32, initial: int = 1500) -> pd.DataFrame:
    """
    Compute ELO ratings across all (dataset, fold) matchups.

    metric_map: {task_type -> metric_column} e.g. {"regression": "r2", "classification": "f1_macro"}
    For each (dataset, fold), every pair of methods is compared; the method with the
    higher metric value wins the matchup (ties are treated as 0.5 each).
    """
    ratings = {m: float(initial) for m in df["method"].unique()}

    for (dataset, fold), grp in df.groupby(["dataset", "fold"]):
        task_type = grp["task_type"].iloc[0]
        metric = metric_map.get(task_type)
        if metric is None or metric not in grp.columns:
            continue

        method_scores = (
            grp[grp["status"].isin(["ok", "cached"])]
            .set_index("method")[metric]
            .dropna()
            .to_dict()
        )
        methods = list(method_scores.keys())

        # All pairs
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                a, b = methods[i], methods[j]
                sa, sb = method_scores[a], method_scores[b]

                if sa > sb:
                    s_a, s_b = 1.0, 0.0
                elif sb > sa:
                    s_a, s_b = 0.0, 1.0
                else:
                    s_a, s_b = 0.5, 0.5

                ra, rb = ratings[a], ratings[b]
                ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
                eb = 1.0 - ea

                ratings[a] = ra + K * (s_a - ea)
                ratings[b] = rb + K * (s_b - eb)

    result = (
        pd.DataFrame({"method": list(ratings.keys()), "elo": list(ratings.values())})
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )
    result["rank"] = result.index + 1
    return result


def _win_rate(df: pd.DataFrame, metric: str, challenger: str, baselines: list[str]) -> pd.DataFrame:
    """Per-dataset win rate: fraction of folds where challenger > baseline."""
    rows = []
    for ds, grp in df.groupby("dataset"):
        for bl in baselines:
            bl_vals = grp[grp["method"] == bl][metric].dropna().values
            ch_vals = grp[grp["method"] == challenger][metric].dropna().values
            n = min(len(bl_vals), len(ch_vals))
            if n == 0:
                continue
            wins = int(np.sum(ch_vals[:n] > bl_vals[:n]))
            rows.append({"dataset": ds, "vs": bl, "wins": wins, "total": n,
                         "win_rate": wins / n,
                         f"{challenger}_{metric}_mean": float(np.mean(ch_vals)),
                         f"{bl}_{metric}_mean": float(np.mean(bl_vals)),
                         f"delta_{metric}": float(np.mean(ch_vals) - np.mean(bl_vals))})
    return pd.DataFrame(rows)


def _regression_fallback_summary(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "task_type",
        "dataset",
        "method",
        "regression_router_fallback_stage",
        "regression_router_fallback_reason",
    }
    if not required.issubset(df.columns):
        return pd.DataFrame()
    reg = df[df["task_type"] == "regression"].copy()
    if reg.empty:
        return pd.DataFrame()
    reg["regression_router_fallback_stage"] = (
        reg["regression_router_fallback_stage"].fillna("missing").astype(str)
    )
    reg["regression_router_fallback_reason"] = (
        reg["regression_router_fallback_reason"].fillna("missing").astype(str)
    )
    summary = (
        reg.groupby(
            [
                "dataset",
                "method",
                "regression_router_fallback_stage",
                "regression_router_fallback_reason",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            ["dataset", "method", "count", "regression_router_fallback_stage", "regression_router_fallback_reason"],
            ascending=[True, True, False, True, True],
        )
        .reset_index(drop=True)
    )
    return summary


def _regression_route_state_summary(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "task_type",
        "dataset",
        "method",
        "router_kind",
        "router_nonfinite_fallback",
        "regression_router_fallback_stage",
        "regression_router_fallback_reason",
        "early_exit",
        "router_skipped",
    }
    if not required.issubset(df.columns):
        return pd.DataFrame()
    reg = df[df["task_type"] == "regression"].copy()
    if reg.empty:
        return pd.DataFrame()

    def classify(row: pd.Series) -> str:
        router_kind = str(row.get("router_kind", "") or "")
        early_exit = bool(row.get("early_exit", False))
        router_skipped = bool(row.get("router_skipped", False))
        nonfinite = bool(row.get("router_nonfinite_fallback", False))
        if router_kind == "legitimacy_gate_anchor_only" or (early_exit and router_skipped):
            return "legitimacy_gate_early_exit"
        if nonfinite or "nonfinite_fallback" in router_kind or router_kind == "router_training_nonfinite_anchor_only":
            return "router_fallback"
        return "clean_routed"

    reg["route_state"] = reg.apply(classify, axis=1)
    reg["regression_router_fallback_stage"] = (
        reg["regression_router_fallback_stage"].fillna("none").astype(str)
    )
    reg["regression_router_fallback_reason"] = (
        reg["regression_router_fallback_reason"].fillna("none").astype(str)
    )
    summary = (
        reg.groupby(
            [
                "dataset",
                "method",
                "route_state",
                "regression_router_fallback_stage",
                "regression_router_fallback_reason",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            ["dataset", "method", "count", "route_state", "regression_router_fallback_stage"],
            ascending=[True, True, False, True, True],
        )
        .reset_index(drop=True)
    )
    return summary


def build_report(all_rows: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / "results_granular.csv", index=False)

    # --- Summary per dataset (mean over folds) ---
    numeric_cols = [c for c in df.columns
                    if c not in ("dataset", "fold", "method", "task_type", "status")
                    and pd.api.types.is_numeric_dtype(df[c])]
    agg = df.groupby(["dataset", "method", "task_type"])[numeric_cols].mean().reset_index()
    agg.to_csv(output_dir / "results_summary.csv", index=False)
    fallback_summary = _regression_fallback_summary(df)
    if not fallback_summary.empty:
        fallback_summary.to_csv(output_dir / "regression_fallback_summary.csv", index=False)
    route_state_summary = _regression_route_state_summary(df)
    if not route_state_summary.empty:
        route_state_summary.to_csv(output_dir / "regression_route_state_summary.csv", index=False)

    lines = [
        "=" * 80,
        "SMART BENCHMARK REPORT",
        "=" * 80,
        "",
    ]

    # Regression table
    reg = agg[agg["task_type"] == "regression"]
    if not reg.empty:
        lines.append("REGRESSION  (mean over folds)")
        lines.append("-" * 80)
        pivot = reg.pivot_table(
            index="dataset", columns="method",
            values=["r2", "rmse", "defer"], aggfunc="mean",
        )
        lines.append(pivot.to_string(float_format="{:.4f}".format))
        lines.append("")

        wr = _win_rate(df[df["task_type"] == "regression"], "r2", "graphdrone", ["tabpfn"])
        if not wr.empty:
            lines.append("Win-rate (GraphDrone R² > TabPFN default, per dataset per fold)")
            lines.append(wr[["dataset", "vs", "win_rate", "delta_r2"]].to_string(index=False,
                float_format="{:.3f}".format))
            lines.append("")

        if not fallback_summary.empty:
            lines.append("Regression Fallback Summary")
            lines.append("-" * 80)
            lines.append(
                fallback_summary[
                    [
                        "dataset",
                        "method",
                        "regression_router_fallback_stage",
                        "regression_router_fallback_reason",
                        "count",
                    ]
                ].to_string(index=False)
            )
            lines.append("")

        if not route_state_summary.empty:
            lines.append("Regression Route State Summary")
            lines.append("-" * 80)
            lines.append(
                route_state_summary[
                    [
                        "dataset",
                        "method",
                        "route_state",
                        "regression_router_fallback_stage",
                        "regression_router_fallback_reason",
                        "count",
                    ]
                ].to_string(index=False)
            )
            lines.append("")

    # Classification table
    clf = agg[agg["task_type"] == "classification"]
    if not clf.empty:
        lines.append("CLASSIFICATION  (mean over folds)")
        lines.append("-" * 80)
        pivot_c = clf.pivot_table(
            index="dataset", columns="method",
            values=["f1_macro", "auc_roc", "pr_auc"], aggfunc="mean",
        )
        lines.append(pivot_c.to_string(float_format="{:.4f}".format))
        lines.append("")

        wr_c = _win_rate(df[df["task_type"] == "classification"], "f1_macro", "graphdrone", ["tabpfn"])
        if not wr_c.empty:
            lines.append("Win-rate (GraphDrone F1 > TabPFN default, per dataset per fold)")
            lines.append(wr_c[["dataset", "vs", "win_rate", "delta_f1_macro"]].to_string(
                index=False, float_format="{:.3f}".format))
            lines.append("")

    # ELO ranking (combined regression + classification)
    ok_df = df[df["status"].isin(["ok", "cached"])].copy()
    if not ok_df.empty:
        elo_df = compute_elo(
            ok_df,
            metric_map={"regression": "r2", "classification": "f1_macro"},
        )
        elo_df.to_csv(output_dir / "elo_ranking.csv", index=False)

        lines.append("ELO RANKING  (K=32, all datasets+folds, regression=R²  classification=F1-macro)")
        lines.append("-" * 80)
        lines.append(elo_df[["rank", "method", "elo"]].to_string(index=False, float_format="{:.1f}".format))
        lines.append("")

        # Per-task-type ELO
        for tt, metric in [("regression", "r2"), ("classification", "f1_macro")]:
            sub = ok_df[ok_df["task_type"] == tt]
            if sub.empty:
                continue
            elo_tt = compute_elo(sub, metric_map={tt: metric})
            lines.append(f"  {tt.capitalize()} ELO  (metric={metric})")
            lines.append("  " + elo_tt[["rank", "method", "elo"]].to_string(index=False,
                float_format="{:.1f}".format).replace("\n", "\n  "))
            lines.append("")

    # Overall
    completed = [r for r in all_rows if r.get("status") in ("ok", "cached")]
    failed    = [r for r in all_rows if r.get("status") == "fail"]
    lines += [
        "=" * 80,
        f"Total tasks: {len(all_rows)}   Completed: {len(completed)}   Failed: {len(failed)}",
        f"Success rate: {100*len(completed)/max(1,len(all_rows)):.1f}%",
        "=" * 80,
    ]

    report_text = "\n".join(lines)
    print("\n" + report_text)
    (output_dir / "report.txt").write_text(report_text)
    print(f"\nOutputs written to {output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Smart benchmark runner (Issue #16)")
    parser.add_argument("--tasks", choices=["regression", "classification", "all", "quick"],
                        default="all", help="Dataset group to run")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Explicit dataset names (overrides --tasks)")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2],
                        help="Fold indices (each = a different random seed for the split)")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--cache-dir", type=Path,
                        default=ROOT / "eval" / "geopoe_cache")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "eval" / "geopoe_benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="Shorthand: 3 datasets, fold 0 only")
    parser.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS,
                        default=DEFAULT_METHODS, help="Subset of methods to run")
    args = parser.parse_args()

    if args.quick:
        dataset_registry = QUICK_DATASETS
        args.folds = [0]
    elif args.datasets:
        dataset_registry = {k: ALL_DATASETS[k] for k in args.datasets if k in ALL_DATASETS}
    elif args.tasks == "regression":
        dataset_registry = REGRESSION_DATASETS
    elif args.tasks == "classification":
        dataset_registry = CLASSIFICATION_DATASETS
    else:
        dataset_registry = ALL_DATASETS

    print("=" * 70)
    print("SMART BENCHMARK RUNNER  (Issue #16 — caching + granular metrics)")
    print("=" * 70)
    print(f"  Datasets  : {list(dataset_registry.keys())}")
    print(f"  Folds     : {args.folds}")
    print(f"  Methods   : {args.methods}")
    print(f"  Cache dir : {args.cache_dir}")
    print(f"  GD version: {GRAPHDRONE_VERSION}")
    print(f"  GD preset : {GRAPHDRONE_PRESET}")
    print()

    all_rows: list[dict] = []
    for ds in dataset_registry:
        for fold in args.folds:
            rows = run_task(ds, fold, args.cache_dir, args.max_samples, args.methods)
            all_rows.extend(rows)

    if all_rows:
        build_report(all_rows, args.output_dir)
        metadata = {
            "graphdrone_version": GRAPHDRONE_VERSION,
            "graphdrone_preset": GRAPHDRONE_PRESET,
            "datasets": list(dataset_registry.keys()),
            "folds": args.folds,
            "methods": args.methods,
            "max_samples": args.max_samples,
        }
        (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()
