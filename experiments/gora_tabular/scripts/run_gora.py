"""
run_gora.py — GoRA-Tabular experiment pipeline runner.

Usage:
  python experiments/gora_tabular/scripts/run_gora.py --dataset california
  python experiments/gora_tabular/scripts/run_gora.py --dataset mnist
  python experiments/gora_tabular/scripts/run_gora.py --dataset both

Architecture:
  GoRA-Tabular — routing inside attention logits:
    logit_{ij}^h = <q^h, k^h>/√d + log(τ_h · Ã_{ij}^{i,h} + ε)
  where Ã is formed by the MoE-routed per-head view mixture.
"""
import sys, argparse, time, json
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
from experiments.gora_tabular.src.views import (
    build_california_views, california_view_features,
    build_mnist_views, mnist_view_features,
)
from experiments.gora_tabular.src.moe_router import MoERouter, UniformRouter, RandomRouter
from experiments.gora_tabular.src.row_transformer import (
    GoraTransformer, StandardTransformer, SingleViewTransformer
)
from experiments.gora_tabular.src.baselines import train_mlp, train_hgbr
from experiments.gora_tabular.src.train import (
    precompute_adj_dense, train_gora, predict_gora, get_device
)
from experiments.gora_tabular.src.eval import (
    score, head_specialisation, bin_metric,
    fig_head_affinity, fig_pi_spread, fig_tau, fig_per_bin, write_report
)

SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED)
D_MODEL = 64; N_HEADS = 4; N_LAYERS = 2; K = 15


def _kappa_bins(kp):
    lo, hi = np.percentile(kp, 33), np.percentile(kp, 67)
    return np.where(kp <= lo, "low", np.where(kp <= hi, "medium", "high"))


# ─────────────────────────────────────────────────────────────────────────────
# California Housing
# ─────────────────────────────────────────────────────────────────────────────

def run_california():
    t0 = time.time(); tag = "california"; task = "regression"; out_dim = 1; nc = 1
    print(f"\n{'='*60}\n  GoRA-Tabular: CALIFORNIA HOUSING\n{'='*60}")

    cal = fetch_california_housing()
    X, y = cal.data.astype(np.float32), cal.target.astype(np.float32)
    X[:, [2, 4]] = np.log1p(X[:, [2, 4]])   # log1p AveRooms, AveOccup
    sc = RobustScaler(); X = sc.fit_transform(X).astype(np.float32)
    N = len(X)
    all_i = np.arange(N)
    tr_i, tmp_i = train_test_split(all_i, test_size=0.30, random_state=SEED)
    va_i, te_i = train_test_split(tmp_i, test_size=0.50, random_state=SEED)
    np.save(ART / f"{tag}_X.npy", X); np.save(ART / f"{tag}_y.npy", y)
    print(f"[CA] N={N} train={len(tr_i)} val={len(va_i)} test={len(te_i)}")

    # Views + observers
    views, vfeats = build_california_views(X, k=K)
    view_tags = list(views.keys())    # FULL, GEO, SOCIO, LOWRANK
    Xpca = PCA(n_components=4, random_state=SEED).fit_transform(X).astype(np.float32)
    print("[CA] Computing observers...")
    g, kp = compute_observers(Xpca, {k: vfeats[k] for k in ["GEO", "SOCIO", "LOWRANK"]}, k=K)
    obs_mean, obs_std = g[tr_i].mean(0), g[tr_i].std(0) + 1e-8
    g_sc = ((g - obs_mean) / obs_std).astype(np.float32)
    kappa_bins = _kappa_bins(kp)
    np.save(ART / f"{tag}_g.npy", g_sc); np.save(ART / f"{tag}_kappa.npy", kp)
    obs_dim = g_sc.shape[1]

    # Precompute adj
    dev = get_device()
    print("[CA] Precomputing adjacency matrices...")
    adjs = precompute_adj_dense(views, N, dev)

    metrics = []; bin_rows = []

    # ── B0: MLP ──────────────────────────────────────────────────────────────
    print("\n[B0] MLP baseline...")
    _, preds_b0, _ = train_mlp(X[tr_i], y[tr_i], X[te_i], y[te_i], out_dim, task)
    metrics.append(score("B0_MLP", y[te_i], preds_b0, task))
    bin_rows += bin_metric(preds_b0, y[te_i], kappa_bins[te_i], task, "B0_MLP")

    # ── B1: HGBR ─────────────────────────────────────────────────────────────
    print("[B1] HGBR baseline...")
    hgbr = train_hgbr(X[tr_i], y[tr_i], X[va_i], y[va_i], task)
    preds_b1 = hgbr.predict(X[te_i]).astype(np.float32)
    metrics.append(score("B1_HGBR", y[te_i], preds_b1, task))
    bin_rows += bin_metric(preds_b1, y[te_i], kappa_bins[te_i], task, "B1_HGBR")

    # ── G0: Standard Transformer (no graph) ───────────────────────────────────
    print("\n[G0] Standard Transformer (no graph bias)...")
    g0 = StandardTransformer(X.shape[1], out_dim, D_MODEL, N_HEADS, N_LAYERS)
    g0 = train_gora(g0, X, g_sc, y, adjs, tr_i, va_i, task, nc,
                    epochs=500, patience=50, lr=3e-4, name="G0_Standard")
    p_g0, _, _, _ = predict_gora(g0, X, g_sc, adjs, te_i, task, dev)
    metrics.append(score("G0_Standard", y[te_i], p_g0, task))
    bin_rows += bin_metric(p_g0, y[te_i], kappa_bins[te_i], task, "G0_Standard")

    # ── G1: Single-view (GEO=idx 1) ───────────────────────────────────────────
    print("\n[G1] Single-view Transformer (GEO)...")
    g1 = SingleViewTransformer(X.shape[1], obs_dim, len(view_tags), out_dim,
                               D_MODEL, N_HEADS, N_LAYERS, fixed_view_idx=1)
    g1 = train_gora(g1, X, g_sc, y, adjs, tr_i, va_i, task, nc,
                    epochs=500, patience=50, lr=3e-4, name="G1_SingleView")
    p_g1, _, _, _ = predict_gora(g1, X, g_sc, adjs, te_i, task, dev)
    metrics.append(score("G1_SingleView", y[te_i], p_g1, task))
    bin_rows += bin_metric(p_g1, y[te_i], kappa_bins[te_i], task, "G1_SingleView")

    # ── G2: GoRA (full pi_{i,h,m}) ────────────────────────────────────────────
    print("\n[G2] GoRA-Tabular (full routing)...")
    g2 = GoraTransformer(X.shape[1], obs_dim, len(view_tags), out_dim,
                         D_MODEL, N_HEADS, N_LAYERS)
    g2 = train_gora(g2, X, g_sc, y, adjs, tr_i, va_i, task, nc,
                    epochs=500, patience=50, lr=3e-4, name="G2_GoRA")
    p_g2, _, pi_g2, tau_g2 = predict_gora(g2, X, g_sc, adjs, te_i, task, dev)
    metrics.append(score("G2_GoRA", y[te_i], p_g2, task))
    bin_rows += bin_metric(p_g2, y[te_i], kappa_bins[te_i], task, "G2_GoRA")
    if pi_g2 is not None: np.save(ART / f"{tag}_pi_G2.npy", pi_g2)

    # ── G3: Uniform pi (ablation) ─────────────────────────────────────────────
    print("\n[G3] Uniform-π ablation...")
    g3 = GoraTransformer(X.shape[1], obs_dim, len(view_tags), out_dim,
                         D_MODEL, N_HEADS, N_LAYERS,
                         router_cls=UniformRouter, router_kwargs={})
    g3 = train_gora(g3, X, g_sc, y, adjs, tr_i, va_i, task, nc,
                    epochs=500, patience=50, lr=3e-4, name="G3_Uniform")
    p_g3, _, _, _ = predict_gora(g3, X, g_sc, adjs, te_i, task, dev)
    metrics.append(score("G3_Uniform", y[te_i], p_g3, task))
    bin_rows += bin_metric(p_g3, y[te_i], kappa_bins[te_i], task, "G3_Uniform")

    # ── G4: Shuffled g ablation ────────────────────────────────────────────────
    print("\n[G4] Shuffled-g ablation...")
    g4 = GoraTransformer(X.shape[1], obs_dim, len(view_tags), out_dim,
                         D_MODEL, N_HEADS, N_LAYERS,
                         router_cls=RandomRouter, router_kwargs={})
    g4 = train_gora(g4, X, g_sc, y, adjs, tr_i, va_i, task, nc,
                    epochs=500, patience=50, lr=3e-4, name="G4_Random")
    p_g4, _, _, _ = predict_gora(g4, X, g_sc, adjs, te_i, task, dev)
    metrics.append(score("G4_Random", y[te_i], p_g4, task))

    # ── Head specialisation ────────────────────────────────────────────────────
    if pi_g2 is not None:
        pi_full_g2 = np.load(ART / f"{tag}_pi_G2.npy")
        spec_df = head_specialisation(pi_full_g2, view_tags)
        spec_df.to_csv(ART / f"{tag}_head_specialisation.csv", index=False)
        print(f"\n[Head Specialisation]\n{spec_df[['head','dominant_view','entropy']].to_string(index=False)}")
    else:
        spec_df = pd.DataFrame()

    # ── Metrics + routing stats ────────────────────────────────────────────────
    pd.DataFrame(metrics).to_csv(ART / f"{tag}_metrics.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(ART / f"{tag}_bin_metrics.csv", index=False)

    # ── Figures ────────────────────────────────────────────────────────────────
    fig_dir = FIG / tag; fig_dir.mkdir(exist_ok=True)
    if pi_g2 is not None and len(spec_df):
        fig_head_affinity(spec_df, view_tags, fig_dir, tag)
        fig_pi_spread(pi_full_g2, view_tags, fig_dir, tag)
    if tau_g2 is not None:
        fig_tau(tau_g2, fig_dir, tag)
    fig_per_bin(bin_rows, task, fig_dir, tag)

    # ── Report ─────────────────────────────────────────────────────────────────
    write_report(tag, task, metrics, spec_df,
                 tau_g2 if tau_g2 is not None else [],
                 view_tags, N_HEADS,
                 REP / f"{tag}_report.md")
    print(f"\n[DONE] California in {time.time()-t0:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# MNIST-784
# ─────────────────────────────────────────────────────────────────────────────

def run_mnist(n_subset=10000):
    t0 = time.time(); tag = "mnist"; task = "classification"; out_dim = 10; nc = 10
    print(f"\n{'='*60}\n  GoRA-Tabular: MNIST-784\n{'='*60}")

    data = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = data.data.astype(np.float32), data.target.astype(np.int64)
    if n_subset and n_subset < len(X):
        sss = StratifiedShuffleSplit(1, train_size=n_subset, random_state=SEED)
        idx, _ = next(sss.split(X, y)); X, y = X[idx], y[idx]
    sc = StandardScaler(); X = sc.fit_transform(X).astype(np.float32)
    N = len(X)
    tr_i, tmp_i = train_test_split(np.arange(N), test_size=0.30, random_state=SEED, stratify=y)
    va_i, te_i = train_test_split(tmp_i, test_size=0.50, random_state=SEED, stratify=y[tmp_i])
    np.save(ART / f"{tag}_X.npy", X); np.save(ART / f"{tag}_y.npy", y)
    print(f"[MN] N={N} train={len(tr_i)} val={len(va_i)} test={len(te_i)}")

    views, vfeats = build_mnist_views(X, k=K)
    view_tags = list(views.keys())    # FULL, BLOCK, PCA
    Xpca = PCA(n_components=50, random_state=SEED).fit_transform(X).astype(np.float32)
    print("[MN] Computing observers...")
    g, kp = compute_observers(Xpca, {k: vfeats[k] for k in ["BLOCK", "PCA"]}, k=K)
    obs_mean, obs_std = g[tr_i].mean(0), g[tr_i].std(0) + 1e-8
    g_sc = ((g - obs_mean) / obs_std).astype(np.float32)
    kappa_bins = _kappa_bins(kp)
    np.save(ART / f"{tag}_g.npy", g_sc); np.save(ART / f"{tag}_kappa.npy", kp)
    obs_dim = g_sc.shape[1]

    dev = get_device()
    print("[MN] Precomputing adjacency matrices...")
    adjs = precompute_adj_dense(views, N, dev)

    metrics = []; bin_rows = []

    # ── B0 MLP ────────────────────────────────────────────────────────────────
    print("\n[B0] MLP..."); _, p_b0, pb0 = train_mlp(X[tr_i], y[tr_i], X[te_i], y[te_i], out_dim, task)
    metrics.append(score("B0_MLP", y[te_i], p_b0, task, pb0))
    bin_rows += bin_metric(p_b0, y[te_i], kappa_bins[te_i], task, "B0_MLP")

    # ── B1 HGBR ───────────────────────────────────────────────────────────────
    print("[B1] HGBR...")
    hgbr = train_hgbr(X[tr_i], y[tr_i], X[va_i], y[va_i], task)
    p_b1 = hgbr.predict(X[te_i]); pb1 = hgbr.predict_proba(X[te_i])
    metrics.append(score("B1_HGBR", y[te_i], p_b1, task, pb1))
    bin_rows += bin_metric(p_b1, y[te_i], kappa_bins[te_i], task, "B1_HGBR")

    # ── G0 Standard ────────────────────────────────────────────────────────────
    print("\n[G0] Standard Transformer...")
    g0 = StandardTransformer(X.shape[1], out_dim, D_MODEL, N_HEADS, N_LAYERS)
    g0 = train_gora(g0, X, g_sc, y, adjs, tr_i, va_i, task, nc,
                    epochs=500, patience=50, lr=3e-4, name="G0_Standard")
    p_g0, pr_g0, _, _ = predict_gora(g0, X, g_sc, adjs, te_i, task, dev)
    metrics.append(score("G0_Standard", y[te_i], p_g0, task, pr_g0))
    bin_rows += bin_metric(p_g0, y[te_i], kappa_bins[te_i], task, "G0_Standard")

    # ── G1 Single-view (PCA=idx 2) ─────────────────────────────────────────────
    print("\n[G1] SingleView (PCA)...")
    g1 = SingleViewTransformer(X.shape[1], obs_dim, len(view_tags), out_dim,
                               D_MODEL, N_HEADS, N_LAYERS, fixed_view_idx=2)
    g1 = train_gora(g1, X, g_sc, y, adjs, tr_i, va_i, task, nc,
                    epochs=500, patience=50, lr=3e-4, name="G1_SingleView")
    p_g1, pr_g1, _, _ = predict_gora(g1, X, g_sc, adjs, te_i, task, dev)
    metrics.append(score("G1_SingleView", y[te_i], p_g1, task, pr_g1))
    bin_rows += bin_metric(p_g1, y[te_i], kappa_bins[te_i], task, "G1_SingleView")

    # ── G2 GoRA ─────────────────────────────────────────────────────────────────
    print("\n[G2] GoRA-Tabular...")
    g2 = GoraTransformer(X.shape[1], obs_dim, len(view_tags), out_dim,
                         D_MODEL, N_HEADS, N_LAYERS)
    g2 = train_gora(g2, X, g_sc, y, adjs, tr_i, va_i, task, nc,
                    epochs=500, patience=50, lr=3e-4, name="G2_GoRA")
    p_g2, pr_g2, pi_g2, tau_g2 = predict_gora(g2, X, g_sc, adjs, te_i, task, dev)
    metrics.append(score("G2_GoRA", y[te_i], p_g2, task, pr_g2))
    bin_rows += bin_metric(p_g2, y[te_i], kappa_bins[te_i], task, "G2_GoRA")
    if pi_g2 is not None: np.save(ART / f"{tag}_pi_G2.npy", pi_g2)

    # ── G3 Uniform ─────────────────────────────────────────────────────────────
    print("\n[G3] Uniform-π ablation...")
    g3 = GoraTransformer(X.shape[1], obs_dim, len(view_tags), out_dim,
                         D_MODEL, N_HEADS, N_LAYERS,
                         router_cls=UniformRouter, router_kwargs={})
    g3 = train_gora(g3, X, g_sc, y, adjs, tr_i, va_i, task, nc,
                    epochs=500, patience=50, lr=3e-4, name="G3_Uniform")
    p_g3, pr_g3, _, _ = predict_gora(g3, X, g_sc, adjs, te_i, task, dev)
    metrics.append(score("G3_Uniform", y[te_i], p_g3, task, pr_g3))
    bin_rows += bin_metric(p_g3, y[te_i], kappa_bins[te_i], task, "G3_Uniform")

    # ── G4 Shuffled ─────────────────────────────────────────────────────────────
    print("\n[G4] Shuffled-g ablation...")
    g4 = GoraTransformer(X.shape[1], obs_dim, len(view_tags), out_dim,
                         D_MODEL, N_HEADS, N_LAYERS,
                         router_cls=RandomRouter, router_kwargs={})
    g4 = train_gora(g4, X, g_sc, y, adjs, tr_i, va_i, task, nc,
                    epochs=500, patience=50, lr=3e-4, name="G4_Random")
    p_g4, pr_g4, _, _ = predict_gora(g4, X, g_sc, adjs, te_i, task, dev)
    metrics.append(score("G4_Random", y[te_i], p_g4, task, pr_g4))

    # ── Head specialisation ─────────────────────────────────────────────────────
    if pi_g2 is not None:
        pi_full = np.load(ART / f"{tag}_pi_G2.npy")
        spec_df = head_specialisation(pi_full, view_tags)
        spec_df.to_csv(ART / f"{tag}_head_specialisation.csv", index=False)
        print(f"\n[Head Specialisation]\n{spec_df[['head','dominant_view','entropy']].to_string(index=False)}")
    else:
        spec_df = pd.DataFrame()

    pd.DataFrame(metrics).to_csv(ART / f"{tag}_metrics.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(ART / f"{tag}_bin_metrics.csv", index=False)

    fig_dir = FIG / tag; fig_dir.mkdir(exist_ok=True)
    if pi_g2 is not None and len(spec_df):
        fig_head_affinity(spec_df, view_tags, fig_dir, tag)
        fig_pi_spread(pi_full, view_tags, fig_dir, tag)
    if tau_g2 is not None:
        fig_tau(tau_g2, fig_dir, tag)
    fig_per_bin(bin_rows, task, fig_dir, tag)

    write_report(tag, task, metrics, spec_df,
                 tau_g2 if tau_g2 is not None else [],
                 view_tags, N_HEADS,
                 REP / f"{tag}_report.md")
    print(f"\n[DONE] MNIST in {time.time()-t0:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["california", "mnist", "both"], default="both")
    p.add_argument("--n-mnist", type=int, default=10000)
    a = p.parse_args()
    if a.dataset in ("california", "both"): run_california()
    if a.dataset in ("mnist", "both"):      run_mnist(a.n_mnist)


if __name__ == "__main__":
    main()
