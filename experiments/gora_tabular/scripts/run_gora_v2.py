"""
run_gora_v2.py — GoRA-Tabular v2: joint-kNN neighbourhood + TabPFN baseline.

New models vs. v1:
  B2 - TabPFN v2 (PriorLabs)
  G5 - GoRA-Tabular with joint-view kNN pool (same arch as G2)
  G6 - GoRA + disagreement-aligned routing loss

Ablation matrix:
  G2  vs G5: topology change only (isolates joint-kNN benefit)
  G5  vs G6: disagreement loss only (isolates routing regularisation)
  G3  vs G3': uniform-pi with joint-kNN (confirms routing > uniform even w/ better topo)

Usage:
  python experiments/gora_tabular/scripts/run_gora_v2.py --dataset california
  python experiments/gora_tabular/scripts/run_gora_v2.py --dataset mnist
  python experiments/gora_tabular/scripts/run_gora_v2.py --dataset both
"""
import sys, argparse, time
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

EXP = Path(__file__).parent.parent
ART = EXP / "artifacts"; ART.mkdir(exist_ok=True)
FIG = EXP / "figures"; FIG.mkdir(exist_ok=True)
REP = EXP / "reports"; REP.mkdir(exist_ok=True)

from experiments.gora_tabular.src.observers import compute_observers
from experiments.gora_tabular.src.views import california_view_features, mnist_view_features
from experiments.gora_tabular.src.moe_router import MoERouter, UniformRouter, RandomRouter
from experiments.gora_tabular.src.row_transformer import (
    GoraTransformer, StandardTransformer, SingleViewTransformer
)
from experiments.gora_tabular.src.baselines import train_mlp, train_hgbr, train_tabpfn
from experiments.gora_tabular.src.train import (
    build_neighbourhood, build_joint_neighbourhood,
    train_gora, predict_gora, get_device
)
from experiments.gora_tabular.src.eval import (
    score, head_specialisation, bin_metric,
    fig_head_affinity, fig_pi_spread, fig_tau, fig_per_bin, write_report_v2
)

SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED)
D_MODEL = 64; N_HEADS = 4; N_LAYERS = 2; K_EACH = 5; K_SINGLE = 15; BATCH = 512
ROUTING_LAM = 0.05   # G6 disagreement loss coefficient


def _kappa_bins(kp):
    lo, hi = np.percentile(kp, 33), np.percentile(kp, 67)
    return np.where(kp <= lo, "low", np.where(kp <= hi, "medium", "high"))


def _train_eval(tag, model, model_name, X, g, y, ni, ew, tr_i, va_i, te_i,
                task, nc, kappa_bins, view_mask=None, agree_score=None, routing_lam=0.0):
    model = train_gora(model, X, g, y, ni, ew, tr_i, va_i, task, nc,
                       epochs=150, patience=20, lr=3e-4, batch_size=BATCH, name=model_name,
                       view_mask=view_mask, agree_score=agree_score, routing_lam=routing_lam)
    p, proba, pi_np, tau_np = predict_gora(model, X, g, y, ni, ew, te_i, task)
    m = score(model_name, y[te_i], p, task, proba)
    br = bin_metric(p, y[te_i], kappa_bins[te_i], task, model_name)
    if pi_np is not None:
        np.save(ART / f"{tag}_pi_{model_name}.npy", pi_np)
    return m, br, pi_np, tau_np


# ─────────────────────────────────────────────────────────────────────────────
# California Housing
# ─────────────────────────────────────────────────────────────────────────────

def run_california():
    t0 = time.time(); tag = "california_v2"; task = "regression"; out_dim = 1; nc = 1
    print(f"\n{'='*60}\n  GoRA-Tabular v2: CALIFORNIA HOUSING\n{'='*60}")

    cal = fetch_california_housing()
    X, y = cal.data.astype(np.float32), cal.target.astype(np.float32)
    X[:, [2, 4]] = np.log1p(X[:, [2, 4]])
    sc = RobustScaler(); X = sc.fit_transform(X).astype(np.float32)
    N = len(X)
    tr_i, tmp_i = train_test_split(np.arange(N), test_size=0.30, random_state=SEED)
    va_i, te_i = train_test_split(tmp_i, test_size=0.50, random_state=SEED)
    print(f"[CA] N={N} train={len(tr_i)} val={len(va_i)} test={len(te_i)}")

    vfeats = california_view_features(X)
    view_tags = list(vfeats.keys()); M = len(view_tags)

    Xpca = PCA(n_components=4, random_state=SEED).fit_transform(X).astype(np.float32)
    print("[CA] Computing observers...")
    g, kp = compute_observers(Xpca, {k: vfeats[k] for k in ["GEO","SOCIO","LOWRANK"]}, k=K_SINGLE)
    obs_mean, obs_std = g[tr_i].mean(0), g[tr_i].std(0) + 1e-8
    g_sc = ((g - obs_mean) / obs_std).astype(np.float32)
    obs_dim = g_sc.shape[1]; kappa_bins = _kappa_bins(kp)

    # V1 neighbourhood (single primary = GEO) — for G2 replication
    print("[CA] Building v1 neighbourhood (single GEO primary)...")
    ni_v1, ew_v1 = build_neighbourhood(vfeats, k=K_SINGLE, primary_key="GEO")

    # V2 neighbourhood (joint-view union)
    print("[CA] Building v2 joint neighbourhood...")
    ni_v2, ew_v2, vm_v2, ag_v2 = build_joint_neighbourhood(vfeats, k_per_view=K_EACH)
    P = ni_v2.shape[1]
    print(f"[CA] Joint pool size P={P} (M={M} × k_each={K_EACH})")

    metrics, bin_rows, spec_rows = [], [], {}

    # ── B0 MLP ────────────────────────────────────────────────────────────────
    print("\n[B0] MLP..."); _, p_b0, _ = train_mlp(X[tr_i], y[tr_i], X[te_i], y[te_i], 1, task)
    metrics.append(score("B0_MLP", y[te_i], p_b0, task))
    bin_rows += bin_metric(p_b0, y[te_i], kappa_bins[te_i], task, "B0_MLP")

    # ── B1 HGBR ───────────────────────────────────────────────────────────────
    print("[B1] HGBR..."); hgbr = train_hgbr(X[tr_i], y[tr_i], X[va_i], y[va_i], task)
    p_b1 = hgbr.predict(X[te_i]).astype(np.float32)
    metrics.append(score("B1_HGBR", y[te_i], p_b1, task))
    bin_rows += bin_metric(p_b1, y[te_i], kappa_bins[te_i], task, "B1_HGBR")

    # ── B2 TabPFN ─────────────────────────────────────────────────────────────
    print("[B2] TabPFN (subsampled to 8k)...")
    try:
        p_b2, _ = train_tabpfn(X[tr_i], y[tr_i], X[te_i], y[te_i], task,
                                max_train_samples=8000, pca_features=False)
        metrics.append(score("B2_TabPFN", y[te_i], p_b2, task))
        bin_rows += bin_metric(p_b2, y[te_i], kappa_bins[te_i], task, "B2_TabPFN")
    except Exception as e:
        print(f"  [TabPFN] FAILED: {e}"); metrics.append({"model": "B2_TabPFN", "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")})

    # ── G2 GoRA v1 (single-primary, for direct comparison) ────────────────────
    print("\n[G2] GoRA-Tabular v1 (GEO-primary neighbourhood)...")
    m, br, pi_g2, tau_g2 = _train_eval(tag, GoraTransformer(X.shape[1], obs_dim, M, out_dim, D_MODEL, N_HEADS, N_LAYERS),
                                        "G2_GoRA_v1", X, g_sc, y, ni_v1, ew_v1, tr_i, va_i, te_i, task, nc, kappa_bins)
    metrics.append(m); bin_rows += br
    if pi_g2 is not None: spec_rows["G2_GoRA_v1"] = head_specialisation(pi_g2, view_tags)

    # ── G5 GoRA v2 (joint neighbourhood, same router, no disagree loss) ────────
    print("\n[G5] GoRA-Tabular v2 (JOINT neighbourhood, no routing loss)...")
    m, br, pi_g5, tau_g5 = _train_eval(
        tag, GoraTransformer(X.shape[1], obs_dim, M, out_dim, D_MODEL, N_HEADS, N_LAYERS),
        "G5_Joint", X, g_sc, y, ni_v2, ew_v2, tr_i, va_i, te_i, task, nc, kappa_bins)
    metrics.append(m); bin_rows += br
    if pi_g5 is not None: spec_rows["G5_Joint"] = head_specialisation(pi_g5, view_tags)

    # ── G6 GoRA v2 + routing disagreement loss ─────────────────────────────────
    print(f"\n[G6] GoRA-Tabular v2 + disagreement routing loss (lam={ROUTING_LAM})...")
    m, br, pi_g6, tau_g6 = _train_eval(
        tag, GoraTransformer(X.shape[1], obs_dim, M, out_dim, D_MODEL, N_HEADS, N_LAYERS),
        "G6_Joint_Reg", X, g_sc, y, ni_v2, ew_v2, tr_i, va_i, te_i, task, nc, kappa_bins,
        view_mask=vm_v2, agree_score=ag_v2, routing_lam=ROUTING_LAM)
    metrics.append(m); bin_rows += br
    if pi_g6 is not None: spec_rows["G6_Joint_Reg"] = head_specialisation(pi_g6, view_tags)

    # ── G3' Uniform-pi with joint kNN ─────────────────────────────────────────
    print("\n[G3'] Uniform-pi + joint kNN...")
    m, br, _, _ = _train_eval(
        tag, GoraTransformer(X.shape[1], obs_dim, M, out_dim, D_MODEL, N_HEADS, N_LAYERS,
                              router_cls=UniformRouter, router_kwargs={}),
        "G3p_Uniform_Joint", X, g_sc, y, ni_v2, ew_v2, tr_i, va_i, te_i, task, nc, kappa_bins)
    metrics.append(m); bin_rows += br

    # ── Outputs ───────────────────────────────────────────────────────────────
    pd.DataFrame(metrics).to_csv(ART/f"{tag}_metrics.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(ART/f"{tag}_bin_metrics.csv", index=False)
    fig_dir = FIG/tag; fig_dir.mkdir(exist_ok=True)
    for mname, sp in spec_rows.items():
        sp.to_csv(ART/f"{tag}_head_spec_{mname}.csv", index=False)
        if len(sp): fig_head_affinity(sp, view_tags, fig_dir, f"{tag}_{mname}")
    if pi_g5 is not None: fig_pi_spread(pi_g5, view_tags, fig_dir, f"{tag}_G5")
    if pi_g6 is not None: fig_pi_spread(pi_g6, view_tags, fig_dir, f"{tag}_G6")
    if tau_g5 is not None: fig_tau(tau_g5, fig_dir, f"{tag}_G5")
    fig_per_bin(bin_rows, task, fig_dir, tag)

    # Agreement score figure
    _fig_agree_score(ag_v2, kp, fig_dir, tag)

    write_report_v2(tag, task, metrics, spec_rows, tau_g5, ag_v2, view_tags, N_HEADS,
                    REP/f"{tag}_report.md")
    print(f"\n[DONE] California v2 in {time.time()-t0:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# MNIST-784
# ─────────────────────────────────────────────────────────────────────────────

def run_mnist(n_subset=10000):
    t0 = time.time(); tag = "mnist_v2"; task = "classification"; out_dim = 10; nc = 10
    print(f"\n{'='*60}\n  GoRA-Tabular v2: MNIST-784\n{'='*60}")

    data = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = data.data.astype(np.float32), data.target.astype(np.int64)
    if n_subset and n_subset < len(X):
        sss = StratifiedShuffleSplit(1, train_size=n_subset, random_state=SEED)
        idx, _ = next(sss.split(X, y)); X, y = X[idx], y[idx]
    sc = StandardScaler(); X = sc.fit_transform(X).astype(np.float32)
    N = len(X)
    tr_i, tmp_i = train_test_split(np.arange(N), test_size=0.30, random_state=SEED, stratify=y)
    va_i, te_i = train_test_split(tmp_i, test_size=0.50, random_state=SEED, stratify=y[tmp_i])
    print(f"[MN] N={N} train={len(tr_i)} val={len(va_i)} test={len(te_i)}")

    vfeats = mnist_view_features(X)
    view_tags = list(vfeats.keys()); M = len(view_tags)

    Xpca50 = PCA(n_components=50, random_state=SEED).fit_transform(X).astype(np.float32)
    print("[MN] Computing observers...")
    g, kp = compute_observers(Xpca50, {k: vfeats[k] for k in ["BLOCK","PCA"]}, k=K_SINGLE)
    obs_mean, obs_std = g[tr_i].mean(0), g[tr_i].std(0) + 1e-8
    g_sc = ((g - obs_mean) / obs_std).astype(np.float32)
    obs_dim = g_sc.shape[1]; kappa_bins = _kappa_bins(kp)

    # V1 neighbourhood (PCA primary)
    print("[MN] Building v1 neighbourhood...")
    ni_v1, ew_v1 = build_neighbourhood(vfeats, k=K_SINGLE, primary_key="PCA")

    # V2 joint neighbourhood
    print("[MN] Building v2 joint neighbourhood...")
    ni_v2, ew_v2, vm_v2, ag_v2 = build_joint_neighbourhood(vfeats, k_per_view=K_EACH)
    P = ni_v2.shape[1]
    print(f"[MN] Joint pool size P={P} (M={M} × k_each={K_EACH})")

    metrics, bin_rows, spec_rows = [], [], {}

    # Baselines
    print("\n[B0] MLP..."); _, p_b0, pb0 = train_mlp(X[tr_i], y[tr_i], X[te_i], y[te_i], out_dim, task)
    metrics.append(score("B0_MLP", y[te_i], p_b0, task, pb0))
    bin_rows += bin_metric(p_b0, y[te_i], kappa_bins[te_i], task, "B0_MLP")

    print("[B1] HGBR..."); hgbr = train_hgbr(X[tr_i], y[tr_i], X[va_i], y[va_i], task)
    p_b1 = hgbr.predict(X[te_i]); pb1 = hgbr.predict_proba(X[te_i])
    metrics.append(score("B1_HGBR", y[te_i], p_b1, task, pb1))
    bin_rows += bin_metric(p_b1, y[te_i], kappa_bins[te_i], task, "B1_HGBR")

    print("[B2] TabPFN (PCA-200)...")
    try:
        p_b2, pb2 = train_tabpfn(X[tr_i], y[tr_i], X[te_i], y[te_i], task,
                                   max_train_samples=8000, max_features=200, pca_features=True)
        metrics.append(score("B2_TabPFN", y[te_i], p_b2, task, pb2))
        bin_rows += bin_metric(p_b2, y[te_i], kappa_bins[te_i], task, "B2_TabPFN")
    except Exception as e:
        print(f"  [TabPFN] FAILED: {e}"); metrics.append({"model": "B2_TabPFN", "accuracy": float("nan"), "macro_f1": float("nan"), "log_loss": float("nan")})

    # G2 v1
    print("\n[G2] GoRA v1 (PCA-primary)...")
    m, br, pi_g2, tau_g2 = _train_eval(tag, GoraTransformer(X.shape[1], obs_dim, M, out_dim, D_MODEL, N_HEADS, N_LAYERS),
                                        "G2_GoRA_v1", X, g_sc, y, ni_v1, ew_v1, tr_i, va_i, te_i, task, nc, kappa_bins)
    metrics.append(m); bin_rows += br
    if pi_g2 is not None: spec_rows["G2_GoRA_v1"] = head_specialisation(pi_g2, view_tags)

    # G5
    print("\n[G5] GoRA v2 (JOINT neighbourhood)...")
    m, br, pi_g5, tau_g5 = _train_eval(
        tag, GoraTransformer(X.shape[1], obs_dim, M, out_dim, D_MODEL, N_HEADS, N_LAYERS),
        "G5_Joint", X, g_sc, y, ni_v2, ew_v2, tr_i, va_i, te_i, task, nc, kappa_bins)
    metrics.append(m); bin_rows += br
    if pi_g5 is not None: spec_rows["G5_Joint"] = head_specialisation(pi_g5, view_tags)

    # G6
    print(f"\n[G6] GoRA v2 + disagree loss (lam={ROUTING_LAM})...")
    m, br, pi_g6, tau_g6 = _train_eval(
        tag, GoraTransformer(X.shape[1], obs_dim, M, out_dim, D_MODEL, N_HEADS, N_LAYERS),
        "G6_Joint_Reg", X, g_sc, y, ni_v2, ew_v2, tr_i, va_i, te_i, task, nc, kappa_bins,
        view_mask=vm_v2, agree_score=ag_v2, routing_lam=ROUTING_LAM)
    metrics.append(m); bin_rows += br
    if pi_g6 is not None: spec_rows["G6_Joint_Reg"] = head_specialisation(pi_g6, view_tags)

    # G3'
    print("\n[G3'] Uniform-pi + joint kNN...")
    m, br, _, _ = _train_eval(
        tag, GoraTransformer(X.shape[1], obs_dim, M, out_dim, D_MODEL, N_HEADS, N_LAYERS,
                              router_cls=UniformRouter, router_kwargs={}),
        "G3p_Uniform_Joint", X, g_sc, y, ni_v2, ew_v2, tr_i, va_i, te_i, task, nc, kappa_bins)
    metrics.append(m); bin_rows += br

    # Outputs
    pd.DataFrame(metrics).to_csv(ART/f"{tag}_metrics.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(ART/f"{tag}_bin_metrics.csv", index=False)
    fig_dir = FIG/tag; fig_dir.mkdir(exist_ok=True)
    for mname, sp in spec_rows.items():
        sp.to_csv(ART/f"{tag}_head_spec_{mname}.csv", index=False)
        if len(sp): fig_head_affinity(sp, view_tags, fig_dir, f"{tag}_{mname}")
    if pi_g5 is not None: fig_pi_spread(pi_g5, view_tags, fig_dir, f"{tag}_G5")
    if tau_g5 is not None: fig_tau(tau_g5, fig_dir, f"{tag}_G5")
    fig_per_bin(bin_rows, task, fig_dir, tag)
    _fig_agree_score(ag_v2, kp, fig_dir, tag)

    write_report_v2(tag, task, metrics, spec_rows, tau_g5, ag_v2, view_tags, N_HEADS,
                    REP/f"{tag}_report.md")
    print(f"\n[DONE] MNIST v2 in {time.time()-t0:.1f}s")


# ─── Helper figures ───────────────────────────────────────────────────────────

def _fig_agree_score(ag, kp, fig_dir, tag):
    """Scatter: routing agree_score vs. curvature κ — expected correlation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(kp, ag, alpha=0.2, s=4, color="#4C72B0")
    ax.set_xlabel("κ (curvature)"); ax.set_ylabel("View agree_score")
    ax.set_title(f"View agreement vs. curvature — {tag}")
    plt.tight_layout(); plt.savefig(fig_dir / f"agree_vs_kappa_{tag}.png", dpi=150); plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["california", "mnist", "both"], default="both")
    p.add_argument("--n-mnist", type=int, default=10000)
    a = p.parse_args()
    if a.dataset in ("california", "both"): run_california()
    if a.dataset in ("mnist", "both"):      run_mnist(a.n_mnist)


if __name__ == "__main__":
    import argparse
    main()
