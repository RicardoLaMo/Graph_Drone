"""
run_audit.py — Main routing contract audit runner.

Usage:
  python experiments/routing_contract_audit/scripts/run_audit.py --dataset california
  python experiments/routing_contract_audit/scripts/run_audit.py --dataset mnist
  python experiments/routing_contract_audit/scripts/run_audit.py --dataset both

Order of operations (contract §routing_contract_readback.md):
  1. [MANDATORY] Run routing semantic tests — abort if any fail
  2. Load data + build views + compute observers
  3. Train shared view encoders → frozen reps
  4. Train Model A (post-hoc) and Model B (explicit routing)
  5. Evaluate predictions + routing behavior
  6. Generate figures and report
"""

import sys, subprocess, argparse, time
import numpy as np
import torch
import pandas as pd
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

EXP = Path(__file__).parent.parent
ART = EXP / "artifacts"; ART.mkdir(exist_ok=True)
FIG = EXP / "figures"; FIG.mkdir(exist_ok=True)
REP = EXP / "reports"; REP.mkdir(exist_ok=True)

from experiments.routing_contract_audit.src.data_california import load_california
from experiments.routing_contract_audit.src.data_mnist import load_mnist
from experiments.routing_contract_audit.src.views import california_views, mnist_views
from experiments.routing_contract_audit.src.observers import compute_observers, bin_kappa, stability
from experiments.routing_contract_audit.src.train import (
    train_view_encoders, stack_reps, train_model_a, train_model_b
)
from experiments.routing_contract_audit.src.eval import (
    predict_a, predict_b, routing_stats,
    fig_pi_dist, fig_beta_dist, fig_per_bin,
    write_report, score
)

SEED = 42; torch.manual_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Mandatory semantic test checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def run_semantic_tests():
    print("=" * 60)
    print("  STEP 1: ROUTING CONTRACT SEMANTIC TESTS (MANDATORY)")
    print("=" * 60)
    test_file = EXP / "tests" / "test_routing_semantics.py"
    result = subprocess.run(
        [sys.executable, str(test_file), "-v"],
        capture_output=False, text=True
    )
    if result.returncode != 0:
        print("\n[ABORT] Routing semantic tests FAILED. Contract not satisfied.")
        print("Fix the implementation before running dataset experiments.")
        sys.exit(1)
    print("\n[OK] All routing semantic tests passed. Proceeding to dataset audit.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2-6 — Dataset audit
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset(name: str, n_mnist: int = 10000):
    t0 = time.time()
    is_ca = (name == "california")
    task = "regression" if is_ca else "classification"
    n_classes = 1 if is_ca else 10
    rep_dim = 64; obs_dim = 4

    print(f"\n{'='*60}\n  AUDIT: {name.upper()}\n{'='*60}")

    # ── Data ──────────────────────────────────────────────────
    if is_ca:
        X, y, tr_i, va_i, te_i = load_california(ART)
        graphs, Xpca = california_views(X, y)
        view_tags = list(graphs.keys())  # FULL, GEO, SOCIO, LOWRANK
    else:
        X, y, tr_i, va_i, te_i = load_mnist(ART, n_mnist)
        graphs, Xpca = mnist_views(X, y)
        view_tags = list(graphs.keys())  # FULL, BLOCK, PCA

    n_views = len(view_tags)
    print(f"  Views: {view_tags}")

    # ── Observers (routing priors ONLY) ───────────────────────
    print("[obs] Computing observer vector on PCA space...")
    obs, kp = compute_observers(Xpca, k=15)
    stab = stability(Xpca)
    pd.DataFrame([stab]).to_csv(ART/f"{name}_stability.csv", index=False)
    kappa_bins = bin_kappa(kp)
    np.save(ART/f"{name}_kappa.npy", kp)
    np.save(ART/f"{name}_kappa_bins.npy", kappa_bins)
    np.save(ART/f"{name}_obs.npy", obs)
    # Scale observers on train set
    obs_mean = obs[tr_i].mean(0); obs_std = obs[tr_i].std(0) + 1e-8
    obs_sc = ((obs - obs_mean) / obs_std).astype(np.float32)

    # ── Step 3: Shared view encoders ──────────────────────────
    print(f"\n[STEP 3] Training shared view encoders (same for A and B)...")
    all_reps = train_view_encoders(
        graphs, tr_i, va_i, rep_dim=rep_dim,
        task=task, n_classes=n_classes,
        epochs=200, patience=25
    )
    # save reps
    for tag, r in all_reps.items():
        np.save(ART/f"{name}_reps_{tag}.npy", r.numpy())
    reps_t = stack_reps(all_reps)   # [N, V, D]
    obs_t = torch.tensor(obs_sc, dtype=torch.float32)

    # ── Step 4A: Train Model A (post-hoc, no pi/beta) ─────────
    print(f"\n[STEP 4A] Training Model A (post-hoc combiner)...")
    out_dim = 1 if is_ca else n_classes
    model_a = train_model_a(reps_t, obs_t, y, tr_i, va_i,
                             n_views, rep_dim, obs_dim, out_dim, task)
    torch.save(model_a.state_dict(), ART/f"{name}_model_A.pt")

    # ── Step 4B: Train Model B (explicit pi + beta) ────────────
    print(f"\n[STEP 4B] Training Model B (intended router)...")
    model_b = train_model_b(reps_t, obs_t, y, tr_i, va_i,
                             n_views, rep_dim, obs_dim, out_dim, task)
    torch.save(model_b.state_dict(), ART/f"{name}_model_B.pt")

    # ── Step 5: Evaluate + routing diagnostics ────────────────
    print(f"\n[STEP 5] Evaluating...")
    mA, preds_a, proba_a = predict_a(model_a, reps_t, obs_t, te_i, y, task)
    mB, preds_b, proba_b, pi, beta = predict_b(model_b, reps_t, obs_t, te_i, y, task)

    np.save(ART/f"{name}_pi.npy", pi)
    np.save(ART/f"{name}_beta.npy", beta)
    np.save(ART/f"{name}_preds_A.npy", preds_a)
    np.save(ART/f"{name}_preds_B.npy", preds_b)

    metrics = [mA, mB]
    pd.DataFrame(metrics).to_csv(ART/f"{name}_metrics.csv", index=False)

    r_stats = routing_stats(pi, beta, kappa_bins, view_tags, te_i)
    r_stats.to_csv(ART/f"{name}_routing_stats.csv", index=False)
    print("\n[routing stats]"); print(r_stats.to_string(index=False))

    # ── Step 6: Figures + report ──────────────────────────────
    dataset_fig = FIG / name; dataset_fig.mkdir(exist_ok=True)
    fig_pi_dist(pi, te_i, view_tags, dataset_fig)
    fig_beta_dist(beta, te_i, dataset_fig)
    fig_per_bin(preds_a, preds_b, y, te_i, kappa_bins, task, dataset_fig)
    print("[figs] Saved to", dataset_fig)

    write_report(
        dataset_name=name, task=task,
        metrics=metrics, routing_df=r_stats, view_tags=view_tags,
        rep_path=REP/f"{name}_report.md",
    )
    print(f"\n[DONE] {name} audit in {time.time()-t0:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["california","mnist","both"], default="both")
    parser.add_argument("--n-mnist", type=int, default=10000)
    parser.add_argument("--skip-tests", action="store_true",
                        help="DANGEROUS: skip semantic tests (only for debugging)")
    a = parser.parse_args()

    if not a.skip_tests:
        run_semantic_tests()
    else:
        print("[WARNING] --skip-tests set. Semantic tests not run.")

    if a.dataset in ("california", "both"):
        run_dataset("california")
    if a.dataset in ("mnist", "both"):
        run_dataset("mnist", a.n_mnist)


if __name__ == "__main__":
    main()
