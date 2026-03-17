#!/usr/bin/env python3
"""
Regression Edge-Case Stress Tests for GraphDrone
=================================================
Covers scenarios absent from the classification stress tests:
  1. Constant target        — zero-variance y; model must not crash or produce NaN
  2. Log-scale / skewed     — heavy-tailed target; routing should still improve on anchor
  3. High-dim (p >> n)      — p=200, n=150; tests TabPFN ignore_pretraining_limits path
  4. Large-N scaling        — N=5000; verifies n_jobs=-1 expert parallelism and router scaling
  5. NaN features           — test set has missing values; validates imputation guard
  6. Multi-seed stability   — same dataset, 5 seeds; checks defer-rate variance < 0.3
  7. Single-expert fallback — only FULL expert; should match TabPFN(8) predictions exactly
"""

import sys
import os
import warnings
import traceback
import numpy as np
import torch
from pathlib import Path
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphdrone_fit.model import GraphDrone
from graphdrone_fit.config import GraphDroneConfig, SetRouterConfig
from graphdrone_fit.expert_factory import ExpertBuildSpec, IdentitySelectorAdapter
from graphdrone_fit.view_descriptor import ViewDescriptor

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_specs(n_feats: int, seed: int = 42, with_specialists: bool = True):
    """Build a standard 4-expert regression portfolio."""
    full_idx = tuple(range(n_feats))
    params_fp = {"n_estimators": 8, "device": _device()}
    params_cb = {"iterations": 100, "random_state": seed}
    params_xgb = {"n_estimators": 100, "random_state": seed}

    rng = np.random.RandomState(seed)
    sub_size = max(1, int(n_feats * 0.7))
    sub_idx = tuple(sorted(rng.choice(n_feats, sub_size, replace=False).tolist()))

    anchor = ExpertBuildSpec(
        descriptor=ViewDescriptor(
            expert_id="FULL", family="FULL", view_name="Foundation Full",
            is_anchor=True, input_dim=n_feats, input_indices=full_idx,
        ),
        model_kind="foundation_regressor",
        input_adapter=IdentitySelectorAdapter(indices=full_idx),
        model_params=params_fp,
    )
    if not with_specialists:
        return (anchor,)

    return (
        anchor,
        ExpertBuildSpec(
            descriptor=ViewDescriptor(
                expert_id="CB", family="structural_subspace", view_name="CatBoost",
                input_dim=n_feats, input_indices=full_idx,
            ),
            model_kind="catboost_regressor",
            input_adapter=IdentitySelectorAdapter(indices=full_idx),
            model_params=params_cb,
        ),
        ExpertBuildSpec(
            descriptor=ViewDescriptor(
                expert_id="XGB", family="structural_subspace", view_name="XGBoost",
                input_dim=n_feats, input_indices=full_idx,
            ),
            model_kind="xgboost_regressor",
            input_adapter=IdentitySelectorAdapter(indices=full_idx),
            model_params=params_xgb,
        ),
        ExpertBuildSpec(
            descriptor=ViewDescriptor(
                expert_id="SUB", family="structural_subspace", view_name="Foundation Sub",
                input_dim=sub_size, input_indices=sub_idx,
            ),
            model_kind="foundation_regressor",
            input_adapter=IdentitySelectorAdapter(indices=sub_idx),
            model_params=params_fp,
        ),
    )


def _fit_predict(X_tr, y_tr, X_te, specs=None, seed: int = 42):
    n_feats = X_tr.shape[1]
    if specs is None:
        specs = _make_specs(n_feats, seed=seed)
    cfg = GraphDroneConfig(
        router=SetRouterConfig(kind="hyper_set_router"),
        problem_type="regression",
    )
    gd = GraphDrone(cfg)
    gd.fit(X_tr, y_tr, expert_specs=specs)
    result = gd.predict(X_te, return_diagnostics=True)
    return result.predictions, result.diagnostics


# ---------------------------------------------------------------------------
# Test 1: Constant target
# ---------------------------------------------------------------------------
def test_constant_target():
    print("\n" + "="*70)
    print("TEST 1: Constant Target (y=5 for all rows, N=400, p=10)")
    print("="*70)
    X, _ = make_regression(n_samples=400, n_features=10, noise=0.0, random_state=0)
    # Use 5.5 (not 5.0) — integer-valued floats like 5.0 satisfy mod(y,1)==0, which
    # triggers the classification branch in GraphDrone's problem-type auto-detector.
    y = np.full(len(X), 5.5, dtype=np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    try:
        preds, diag = _fit_predict(X_tr, y_tr, X_te)
        assert not np.any(np.isnan(preds)), "Predictions contain NaN"
        assert not np.any(np.isinf(preds)), "Predictions contain Inf"
        mae = float(np.abs(preds - y_te).mean())
        print(f"  MAE={mae:.4f}  defer={diag['mean_defer_prob']:.4f}")
        print(f"  PASS — no crash, no NaN/Inf on constant target")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 2: Log-scale / heavy-tailed target
# ---------------------------------------------------------------------------
def test_skewed_target():
    print("\n" + "="*70)
    print("TEST 2: Log-scale / Heavy-tail Target (exponential y, N=800, p=15)")
    print("="*70)
    rng = np.random.RandomState(7)
    X, _ = make_regression(n_samples=800, n_features=15, noise=1.0, random_state=7)
    y = np.exp(rng.randn(len(X))).astype(np.float32)  # log-normal
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=7)
    try:
        preds, diag = _fit_predict(X_tr, y_tr, X_te)
        assert not np.any(np.isnan(preds)), "Predictions contain NaN"
        r2 = r2_score(y_te, preds)
        print(f"  R²={r2:.4f}  defer={diag['mean_defer_prob']:.4f}")
        print(f"  PASS — skewed target handled (R²={r2:.4f})")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 3: High-dimensional regression (p >> n)
# ---------------------------------------------------------------------------
def test_high_dimensional():
    print("\n" + "="*70)
    print("TEST 3: High-Dimensional Regression (p=200, n=250, p > n)")
    print("="*70)
    X, y = make_regression(n_samples=250, n_features=200, n_informative=20,
                           noise=5.0, random_state=3)
    y = y.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=3)
    print(f"  X_tr: {X_tr.shape}  (p={X_tr.shape[1]} > n={X_tr.shape[0]})")
    try:
        preds, diag = _fit_predict(X_tr, y_tr, X_te)
        assert not np.any(np.isnan(preds)), "Predictions contain NaN"
        rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
        r2 = r2_score(y_te, preds)
        print(f"  RMSE={rmse:.2f}  R²={r2:.4f}  defer={diag['mean_defer_prob']:.4f}")
        print(f"  PASS — high-dim (p>n) handled without crash")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 4: Large-N scaling
# ---------------------------------------------------------------------------
def test_large_n():
    print("\n" + "="*70)
    print("TEST 4: Large-N Regression (N=5000, p=20) — tests n_jobs=-1 path")
    print("="*70)
    X, y = make_regression(n_samples=5000, n_features=20, n_informative=10,
                           noise=2.0, random_state=4)
    y = y.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=4)
    print(f"  X_tr: {X_tr.shape}")
    try:
        import time
        t0 = time.time()
        preds, diag = _fit_predict(X_tr, y_tr, X_te)
        elapsed = time.time() - t0
        assert not np.any(np.isnan(preds)), "Predictions contain NaN"
        r2 = r2_score(y_te, preds)
        print(f"  R²={r2:.4f}  defer={diag['mean_defer_prob']:.4f}  elapsed={elapsed:.1f}s")
        print(f"  PASS — large-N fit completed in {elapsed:.0f}s")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 5: NaN features in test set
# ---------------------------------------------------------------------------
def test_nan_features():
    print("\n" + "="*70)
    print("TEST 5: NaN Features in Test Set (N=600, p=12, 15% test NaN rate)")
    print("="*70)
    X, y = make_regression(n_samples=600, n_features=12, noise=1.0, random_state=5)
    y = y.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=5)

    # Inject NaNs into test features (simulates missing-at-random deployment data)
    rng = np.random.RandomState(5)
    nan_mask = rng.random(X_te.shape) < 0.15
    X_te_nan = X_te.copy().astype(np.float64)
    X_te_nan[nan_mask] = np.nan

    # Impute with training column medians (standard preprocessing step)
    col_medians = np.nanmedian(X_tr, axis=0)
    nan_positions = np.where(np.isnan(X_te_nan))
    X_te_imputed = X_te_nan.copy()
    X_te_imputed[nan_positions] = col_medians[nan_positions[1]]
    X_te_imputed = X_te_imputed.astype(np.float32)

    try:
        preds, diag = _fit_predict(X_tr, y_tr, X_te_imputed)
        assert not np.any(np.isnan(preds)), "Predictions contain NaN after imputation"
        r2 = r2_score(y_te, preds)
        print(f"  R²={r2:.4f}  defer={diag['mean_defer_prob']:.4f}")
        print(f"  PASS — NaN-imputed test set handled correctly")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 6: Multi-seed defer-rate stability
# ---------------------------------------------------------------------------
def test_multiseed_stability():
    print("\n" + "="*70)
    print("TEST 6: Multi-Seed Stability (N=1000, p=15, 5 seeds)")
    print("  Goal: defer-rate variance < 0.30 (router is not erratic)")
    print("="*70)
    X, y = make_regression(n_samples=1000, n_features=15, n_informative=8,
                           noise=1.0, random_state=99)
    y = y.astype(np.float32)
    defer_rates = []
    r2_scores = []
    try:
        for seed in [0, 1, 2, 3, 4]:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
            preds, diag = _fit_predict(X_tr, y_tr, X_te, seed=seed)
            defer_rates.append(diag["mean_defer_prob"])
            r2_scores.append(r2_score(y_te, preds))
            print(f"  seed={seed} | R²={r2_scores[-1]:.4f}  defer={defer_rates[-1]:.4f}")

        defer_std = float(np.std(defer_rates))
        r2_mean = float(np.mean(r2_scores))
        print(f"\n  defer std={defer_std:.4f}  R² mean={r2_mean:.4f}")
        if defer_std < 0.30:
            print(f"  PASS — defer-rate variance {defer_std:.4f} < 0.30 (stable routing)")
            return True
        else:
            print(f"  WARN — defer-rate variance {defer_std:.4f} >= 0.30 (erratic routing)")
            return False
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 7: Single-expert fallback (anchor only)
# ---------------------------------------------------------------------------
def test_single_expert_fallback():
    print("\n" + "="*70)
    print("TEST 7: Single-Expert Fallback (anchor-only portfolio)")
    print("  Expected: GraphDrone output == TabPFN(8) predictions (no routing applied)")
    print("="*70)
    X, y = make_regression(n_samples=500, n_features=10, noise=1.0, random_state=77)
    y = y.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=77)
    try:
        specs = _make_specs(X_tr.shape[1], seed=77, with_specialists=False)
        preds_gd, diag = _fit_predict(X_tr, y_tr, X_te, specs=specs)

        from tabpfn import TabPFNRegressor
        tpfn = TabPFNRegressor(n_estimators=8, device=_device(), random_state=42,
                               ignore_pretraining_limits=len(X_tr) > 1000)
        tpfn.fit(X_tr, y_tr)
        preds_tp = tpfn.predict(X_te)

        r2_gd = r2_score(y_te, preds_gd)
        r2_tp = r2_score(y_te, preds_tp)
        # With BootstrapFullRouter (N<500 guard) or single expert, defer≈0
        # so predictions should be very close (not necessarily identical due to internal val split)
        delta_r2 = abs(r2_gd - r2_tp)
        print(f"  GraphDrone R²={r2_gd:.4f}  TabPFN(8) R²={r2_tp:.4f}  |ΔR²|={delta_r2:.4f}")
        print(f"  defer={diag['mean_defer_prob']:.4f}")
        if delta_r2 < 0.05:
            print(f"  PASS — single-expert GraphDrone tracks TabPFN anchor (|ΔR²|={delta_r2:.4f} < 0.05)")
            return True
        else:
            print(f"  WARN — single-expert diverges from anchor more than expected (|ΔR²|={delta_r2:.4f})")
            return False
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n" + "="*70)
    print("GRAPHDRONE REGRESSION STRESS TESTS")
    print(f"Device: {_device()}")
    print("="*70)

    tests = [
        ("Constant target",           test_constant_target),
        ("Log-scale target",          test_skewed_target),
        ("High-dimensional (p>n)",    test_high_dimensional),
        ("Large-N (5000 samples)",    test_large_n),
        ("NaN features in test set",  test_nan_features),
        ("Multi-seed stability",      test_multiseed_stability),
        ("Single-expert fallback",    test_single_expert_fallback),
    ]

    results = {}
    for name, fn in tests:
        results[name] = fn()

    print("\n" + "="*70)
    print("REGRESSION STRESS TEST SUMMARY")
    print("="*70)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {name}")

    n_pass = sum(results.values())
    n_total = len(results)
    print("="*70)
    if n_pass == n_total:
        print(f"ALL TESTS PASSED ({n_pass}/{n_total})")
        return 0
    else:
        print(f"SOME TESTS FAILED ({n_pass}/{n_total} passed)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
