"""
Smoke test: independent regression and classification engines.

Validates that a fresh install from GitHub delivers two fully independent
engines that auto-dispatch correctly and produce sensible results.

Run from repo root:
    pip install -e .
    python tests/test_smoke_v118.py
"""
from __future__ import annotations
import sys
import numpy as np

# ---------------------------------------------------------------------------
# 0. Import guard — confirm graphdrone_fit is installed (not just PYTHONPATH)
# ---------------------------------------------------------------------------
try:
    import graphdrone_fit
    print(f"[OK] graphdrone_fit imported from: {graphdrone_fit.__file__}")
except ImportError as e:
    print(f"[FAIL] Cannot import graphdrone_fit: {e}")
    print("       Run: pip install -e .  (from repo root)")
    sys.exit(1)

from graphdrone_fit import GraphDrone, GraphDroneConfig, SetRouterConfig

# ---------------------------------------------------------------------------
# 1. Synthetic datasets
# ---------------------------------------------------------------------------
rng = np.random.RandomState(0)

# Regression: Boston-style (500 rows, 10 features, continuous target)
N_REG = 500
X_reg = rng.randn(N_REG, 10).astype(np.float32)
y_reg = (X_reg @ rng.randn(10) + rng.randn(N_REG) * 0.5).astype(np.float32)

# Binary classification: XOR-style (400 rows, 8 features, 2 classes)
N_BIN = 400
X_bin = rng.randn(N_BIN, 8).astype(np.float32)
y_bin = ((X_bin[:, 0] * X_bin[:, 1]) > 0).astype(np.int64)

# Multiclass: 3-class (300 rows, 6 features)
N_MC = 300
X_mc = rng.randn(N_MC, 6).astype(np.float32)
y_mc = (np.abs(X_mc[:, 0]) * 3).astype(int).clip(0, 2)

PASSED = []
FAILED = []

def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  [OK]   {name}" + (f"  — {detail}" if detail else ""))
        PASSED.append(name)
    else:
        print(f"  [FAIL] {name}" + (f"  — {detail}" if detail else ""))
        FAILED.append(name)


# ---------------------------------------------------------------------------
# 2. REGRESSION ENGINE
# ---------------------------------------------------------------------------
print("\n── Regression engine (v1-width GORA) ─────────────────────────────")

split = int(N_REG * 0.8)
gd_reg = GraphDrone(GraphDroneConfig())
gd_reg.fit(X_reg[:split], y_reg[:split], problem_type="regression")
result_reg = gd_reg.predict(X_reg[split:], return_diagnostics=True)

preds_reg = result_reg.predictions
check("regression: output shape", preds_reg.shape == (N_REG - split,),
      f"shape={preds_reg.shape}")
check("regression: output dtype float", np.issubdtype(preds_reg.dtype, np.floating))
check("regression: no NaN/Inf", np.all(np.isfinite(preds_reg)))

ss_res = np.sum((y_reg[split:] - preds_reg) ** 2)
ss_tot = np.sum((y_reg[split:] - y_reg[split:].mean()) ** 2)
r2 = 1 - ss_res / ss_tot
check("regression: R² > 0.5", r2 > 0.5, f"R²={r2:.3f}")

# Confirm regression engine diagnostics (GORA router, not geo_poe)
diag_reg = result_reg.diagnostics
check("regression: router_kind is NOT geo_poe",
      diag_reg.get("router_kind") != "geo_poe",
      f"router_kind={diag_reg.get('router_kind')}")
check("regression: problem_type stored as regression",
      gd_reg._problem_type == "regression")


# ---------------------------------------------------------------------------
# 3. CLASSIFICATION ENGINE — binary
# ---------------------------------------------------------------------------
print("\n── Classification engine — binary (multi-view static GeoPOE) ──────")

split_b = int(N_BIN * 0.8)
gd_bin = GraphDrone(GraphDroneConfig(n_classes=2))
gd_bin.fit(X_bin[:split_b], y_bin[:split_b])
result_bin = gd_bin.predict(X_bin[split_b:], return_diagnostics=True)

proba_bin = result_bin.predictions
check("binary clf: output shape [N,2]", proba_bin.shape == (N_BIN - split_b, 2),
      f"shape={proba_bin.shape}")
check("binary clf: valid probabilities (sum≈1)",
      np.allclose(proba_bin.sum(axis=1), 1.0, atol=1e-4))
check("binary clf: no NaN/Inf", np.all(np.isfinite(proba_bin)))

preds_bin = np.argmax(proba_bin, axis=1)
acc_bin = (preds_bin == y_bin[split_b:]).mean()
check("binary clf: accuracy > 0.5", acc_bin > 0.5, f"acc={acc_bin:.3f}")

diag_bin = result_bin.diagnostics
check("binary clf: router_kind=geo_poe (static path)",
      diag_bin.get("router_kind") == "geo_poe",
      f"router_kind={diag_bin.get('router_kind')}")
check("binary clf: problem_type stored as classification",
      gd_bin._problem_type == "classification")


# ---------------------------------------------------------------------------
# 4. CLASSIFICATION ENGINE — multiclass
# ---------------------------------------------------------------------------
print("\n── Classification engine — multiclass (3-class) ───────────────────")

split_m = int(N_MC * 0.8)
gd_mc = GraphDrone(GraphDroneConfig(n_classes=3))
gd_mc.fit(X_mc[:split_m], y_mc[:split_m])
result_mc = gd_mc.predict(X_mc[split_m:], return_diagnostics=True)

proba_mc = result_mc.predictions
check("multiclass: output shape [N,3]", proba_mc.shape == (N_MC - split_m, 3),
      f"shape={proba_mc.shape}")
check("multiclass: valid probabilities (sum≈1)",
      np.allclose(proba_mc.sum(axis=1), 1.0, atol=1e-4))
check("multiclass: no NaN/Inf", np.all(np.isfinite(proba_mc)))
check("multiclass: router_kind=geo_poe",
      result_mc.diagnostics.get("router_kind") == "geo_poe")


# ---------------------------------------------------------------------------
# 5. Auto-dispatch: no problem_type hint — engine infers from y
# ---------------------------------------------------------------------------
print("\n── Auto-dispatch (no problem_type hint) ───────────────────────────")

gd_auto_reg = GraphDrone(GraphDroneConfig())
gd_auto_reg.fit(X_reg[:split], y_reg[:split])          # float y → regression
check("auto-dispatch: float y → regression",
      gd_auto_reg._problem_type == "regression")

gd_auto_clf = GraphDrone(GraphDroneConfig())
gd_auto_clf.fit(X_bin[:split_b], y_bin[:split_b])      # int y, 2 classes → classification
check("auto-dispatch: int y (2 classes) → classification",
      gd_auto_clf._problem_type == "classification")

# Confirm engines are independent: fitting one does not affect the other
check("engine independence: reg instance still regression",
      gd_reg._problem_type == "regression")
check("engine independence: bin instance still classification",
      gd_bin._problem_type == "classification")


# ---------------------------------------------------------------------------
# 6. Expert portfolio check
# ---------------------------------------------------------------------------
print("\n── Expert portfolio composition ────────────────────────────────────")

reg_ids  = set(gd_reg._portfolio.expert_order)
bin_ids  = set(gd_bin._portfolio.expert_order)

check("regression: has FULL expert",  any("FULL" in e for e in reg_ids),
      f"ids={reg_ids}")
check("classification: has FULL expert", any("FULL" in e for e in bin_ids),
      f"ids={bin_ids}")
check("classification: has 3 SUB experts",
      sum(1 for e in bin_ids if "SUB" in e) == 3,
      f"SUB experts={[e for e in bin_ids if 'SUB' in e]}")
check("regression: no SUB3 experts",     # regression keeps v1-width (1 SUB or none)
      sum(1 for e in reg_ids if "SUB2" in e) == 0,
      f"reg ids={reg_ids}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total = len(PASSED) + len(FAILED)
print(f"\n{'='*60}")
print(f"  Passed: {len(PASSED)}/{total}")
if FAILED:
    print(f"  FAILED: {FAILED}")
    print(f"{'='*60}")
    sys.exit(1)
else:
    print(f"  All checks passed — GraphDrone v1.18 ready")
    print(f"{'='*60}")
    sys.exit(0)
