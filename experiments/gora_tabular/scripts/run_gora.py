"""
run_gora.py — GoRA-Tabular experiment pipeline (sparse neighbourhood version).

Usage:
  python experiments/gora_tabular/scripts/run_gora.py --dataset california
  python experiments/gora_tabular/scripts/run_gora.py --dataset mnist
  python experiments/gora_tabular/scripts/run_gora.py --dataset both
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
from experiments.gora_tabular.src.views import (
    california_view_features, mnist_view_features
)
from experiments.gora_tabular.src.moe_router import MoERouter, UniformRouter, RandomRouter
from experiments.gora_tabular.src.row_transformer import (
    GoraTransformer, StandardTransformer, SingleViewTransformer
)
from experiments.gora_tabular.src.baselines import train_mlp, train_hgbr
from experiments.gora_tabular.src.train import (
    build_neighbourhood, train_gora, predict_gora, get_device
)
from experiments.gora_tabular.src.eval import (
    score, head_specialisation, bin_metric,
    fig_head_affinity, fig_pi_spread, fig_tau, fig_per_bin, write_report
)

SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED)
D_MODEL = 64; N_HEADS = 4; N_LAYERS = 2; K = 15; BATCH = 512


def _kappa_bins(kp):
    lo, hi = np.percentile(kp, 33), np.percentile(kp, 67)
    return np.where(kp <= lo, "low", np.where(kp <= hi, "medium", "high"))


def _train_eval(tag, model, model_name, X, g, y, ni, ew, tr_i, va_i, te_i,
                task, nc, kappa_bins):
    model = train_gora(model, X, g, y, ni, ew, tr_i, va_i, task, nc,
                       epochs=150, patience=20, lr=3e-4, batch_size=BATCH, name=model_name)
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
    t0 = time.time(); tag = "california"; task = "regression"; out_dim = 1; nc = 1
    print(f"\n{'='*60}\n  GoRA-Tabular: CALIFORNIA HOUSING\n{'='*60}")

    cal = fetch_california_housing()
    X, y = cal.data.astype(np.float32), cal.target.astype(np.float32)
    X[:, [2, 4]] = np.log1p(X[:, [2, 4]])
    sc = RobustScaler(); X = sc.fit_transform(X).astype(np.float32)
    N = len(X)
    tr_i, tmp_i = train_test_split(np.arange(N), test_size=0.30, random_state=SEED)
    va_i, te_i = train_test_split(tmp_i, test_size=0.50, random_state=SEED)
    print(f"[CA] N={N} train={len(tr_i)} val={len(va_i)} test={len(te_i)}")

    vfeats = california_view_features(X)
    view_tags = list(vfeats.keys())   # FULL, GEO, SOCIO, LOWRANK
    M = len(view_tags)

    Xpca = PCA(n_components=4, random_state=SEED).fit_transform(X).astype(np.float32)
    print("[CA] Computing observers...")
    g, kp = compute_observers(Xpca, {k: vfeats[k] for k in ["GEO","SOCIO","LOWRANK"]}, k=K)
    obs_mean, obs_std = g[tr_i].mean(0), g[tr_i].std(0) + 1e-8
    g_sc = ((g - obs_mean) / obs_std).astype(np.float32)
    obs_dim = g_sc.shape[1]; kappa_bins = _kappa_bins(kp)
    np.save(ART/f"{tag}_g.npy", g_sc); np.save(ART/f"{tag}_kappa.npy", kp)

    print("[CA] Building neighbourhood + edge weights...")
    # Primary view for neighbourhood: GEO (geography dominates CA housing)
    ni, ew = build_neighbourhood(vfeats, k=K, primary_key="GEO")
    np.save(ART/f"{tag}_neigh_idx.npy", ni); np.save(ART/f"{tag}_edge_wts.npy", ew)

    metrics, bin_rows, spec_df_g2, tau_g2 = [], [], None, None

    # ── B0 MLP ────────────────────────────────────────────────────────────────
    print("\n[B0] MLP..."); _, p_b0, _ = train_mlp(X[tr_i], y[tr_i], X[te_i], y[te_i], 1, task)
    metrics.append(score("B0_MLP", y[te_i], p_b0, task)); bin_rows += bin_metric(p_b0, y[te_i], kappa_bins[te_i], task, "B0_MLP")

    # ── B1 HGBR ───────────────────────────────────────────────────────────────
    print("[B1] HGBR..."); hgbr = train_hgbr(X[tr_i], y[tr_i], X[va_i], y[va_i], task)
    p_b1 = hgbr.predict(X[te_i]).astype(np.float32)
    metrics.append(score("B1_HGBR", y[te_i], p_b1, task)); bin_rows += bin_metric(p_b1, y[te_i], kappa_bins[te_i], task, "B1_HGBR")

    # ── G0 Standard ───────────────────────────────────────────────────────────
    print("\n[G0] Standard Transformer (no graph bias)...")
    m, br, _, _ = _train_eval(tag, StandardTransformer(X.shape[1], obs_dim, M, out_dim, D_MODEL, N_HEADS, N_LAYERS),
                               "G0_Standard", X, g_sc, y, ni, ew, tr_i, va_i, te_i, task, nc, kappa_bins)
    metrics.append(m); bin_rows += br

    # ── G1 SingleView (GEO=1) ─────────────────────────────────────────────────
    print("\n[G1] Single-view (GEO)...")
    m, br, _, _ = _train_eval(tag, SingleViewTransformer(X.shape[1], obs_dim, M, out_dim, D_MODEL, N_HEADS, N_LAYERS, fixed_view_idx=1),
                               "G1_SingleView", X, g_sc, y, ni, ew, tr_i, va_i, te_i, task, nc, kappa_bins)
    metrics.append(m); bin_rows += br

    # ── G2 GoRA ───────────────────────────────────────────────────────────────
    print("\n[G2] GoRA-Tabular (full routing)...")
    m, br, pi_g2, tau_np = _train_eval(tag, GoraTransformer(X.shape[1], obs_dim, M, out_dim, D_MODEL, N_HEADS, N_LAYERS),
                                        "G2_GoRA", X, g_sc, y, ni, ew, tr_i, va_i, te_i, task, nc, kappa_bins)
    metrics.append(m); bin_rows += br; tau_g2 = tau_np
    if pi_g2 is not None: spec_df_g2 = head_specialisation(pi_g2, view_tags)

    # ── G3 Uniform ────────────────────────────────────────────────────────────
    print("\n[G3] Uniform-pi ablation...")
    m, br, _, _ = _train_eval(tag, GoraTransformer(X.shape[1], obs_dim, M, out_dim, D_MODEL, N_HEADS, N_LAYERS,
                                                     router_cls=UniformRouter, router_kwargs={}),
                               "G3_Uniform", X, g_sc, y, ni, ew, tr_i, va_i, te_i, task, nc, kappa_bins)
    metrics.append(m); bin_rows += br

    # ── G4 Shuffled ───────────────────────────────────────────────────────────
    print("\n[G4] Shuffled-g ablation...")
    m, br, _, _ = _train_eval(tag, GoraTransformer(X.shape[1], obs_dim, M, out_dim, D_MODEL, N_HEADS, N_LAYERS,
                                                     router_cls=RandomRouter, router_kwargs={}),
                               "G4_Random", X, g_sc, y, ni, ew, tr_i, va_i, te_i, task, nc, kappa_bins)
    metrics.append(m); bin_rows += br

    # ── Outputs ───────────────────────────────────────────────────────────────
    pd.DataFrame(metrics).to_csv(ART/f"{tag}_metrics.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(ART/f"{tag}_bin_metrics.csv", index=False)
    sp = spec_df_g2 if spec_df_g2 is not None else pd.DataFrame()
    if len(sp): sp.to_csv(ART/f"{tag}_head_specialisation.csv", index=False); print(f"\n[Head Spec]\n{sp[['head_idx','dominant_view','entropy']].to_string(index=False)}")
    fig_dir = FIG/tag; fig_dir.mkdir(exist_ok=True)
    if pi_g2 is not None and len(sp): fig_head_affinity(sp, view_tags, fig_dir, tag); fig_pi_spread(pi_g2, view_tags, fig_dir, tag)
    if tau_g2 is not None: fig_tau(tau_g2, fig_dir, tag)
    fig_per_bin(bin_rows, task, fig_dir, tag)
    write_report(tag, task, metrics, sp, tau_g2 if tau_g2 is not None else [], view_tags, N_HEADS, REP/f"{tag}_report.md")
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
    print(f"[MN] N={N} train={len(tr_i)} val={len(va_i)} test={len(te_i)}")

    vfeats = mnist_view_features(X)
    view_tags = list(vfeats.keys())   # FULL, BLOCK, PCA
    M = len(view_tags)

    Xpca = PCA(n_components=50, random_state=SEED).fit_transform(X).astype(np.float32)
    print("[MN] Computing observers...")
    g, kp = compute_observers(Xpca, {k: vfeats[k] for k in ["BLOCK","PCA"]}, k=K)
    obs_mean, obs_std = g[tr_i].mean(0), g[tr_i].std(0) + 1e-8
    g_sc = ((g - obs_mean) / obs_std).astype(np.float32)
    obs_dim = g_sc.shape[1]; kappa_bins = _kappa_bins(kp)
    np.save(ART/f"{tag}_g.npy", g_sc); np.save(ART/f"{tag}_kappa.npy", kp)

    print("[MN] Building neighbourhood + edge weights...")
    ni, ew = build_neighbourhood(vfeats, k=K, primary_key="PCA")
    np.save(ART/f"{tag}_neigh_idx.npy", ni); np.save(ART/f"{tag}_edge_wts.npy", ew)

    metrics, bin_rows, spec_df_g2, tau_g2 = [], [], None, None

    print("\n[B0] MLP..."); _, p_b0, pb0 = train_mlp(X[tr_i], y[tr_i], X[te_i], y[te_i], out_dim, task)
    metrics.append(score("B0_MLP", y[te_i], p_b0, task, pb0)); bin_rows += bin_metric(p_b0, y[te_i], kappa_bins[te_i], task, "B0_MLP")

    print("[B1] HGBR..."); hgbr = train_hgbr(X[tr_i], y[tr_i], X[va_i], y[va_i], task)
    p_b1 = hgbr.predict(X[te_i]); pb1 = hgbr.predict_proba(X[te_i])
    metrics.append(score("B1_HGBR", y[te_i], p_b1, task, pb1)); bin_rows += bin_metric(p_b1, y[te_i], kappa_bins[te_i], task, "B1_HGBR")

    print("\n[G0] Standard Transformer...")
    m, br, _, _ = _train_eval(tag, StandardTransformer(X.shape[1], obs_dim, M, out_dim, D_MODEL, N_HEADS, N_LAYERS),
                               "G0_Standard", X, g_sc, y, ni, ew, tr_i, va_i, te_i, task, nc, kappa_bins)
    metrics.append(m); bin_rows += br

    print("\n[G1] Single-view (PCA=2)...")
    m, br, _, _ = _train_eval(tag, SingleViewTransformer(X.shape[1], obs_dim, M, out_dim, D_MODEL, N_HEADS, N_LAYERS, fixed_view_idx=2),
                               "G1_SingleView", X, g_sc, y, ni, ew, tr_i, va_i, te_i, task, nc, kappa_bins)
    metrics.append(m); bin_rows += br

    print("\n[G2] GoRA-Tabular...")
    m, br, pi_g2, tau_np = _train_eval(tag, GoraTransformer(X.shape[1], obs_dim, M, out_dim, D_MODEL, N_HEADS, N_LAYERS),
                                        "G2_GoRA", X, g_sc, y, ni, ew, tr_i, va_i, te_i, task, nc, kappa_bins)
    metrics.append(m); bin_rows += br; tau_g2 = tau_np
    if pi_g2 is not None: spec_df_g2 = head_specialisation(pi_g2, view_tags)

    print("\n[G3] Uniform-pi ablation...")
    m, br, _, _ = _train_eval(tag, GoraTransformer(X.shape[1], obs_dim, M, out_dim, D_MODEL, N_HEADS, N_LAYERS,
                                                     router_cls=UniformRouter, router_kwargs={}),
                               "G3_Uniform", X, g_sc, y, ni, ew, tr_i, va_i, te_i, task, nc, kappa_bins)
    metrics.append(m); bin_rows += br

    print("\n[G4] Shuffled-g ablation...")
    m, br, _, _ = _train_eval(tag, GoraTransformer(X.shape[1], obs_dim, M, out_dim, D_MODEL, N_HEADS, N_LAYERS,
                                                     router_cls=RandomRouter, router_kwargs={}),
                               "G4_Random", X, g_sc, y, ni, ew, tr_i, va_i, te_i, task, nc, kappa_bins)
    metrics.append(m); bin_rows += br

    pd.DataFrame(metrics).to_csv(ART/f"{tag}_metrics.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(ART/f"{tag}_bin_metrics.csv", index=False)
    sp = spec_df_g2 if spec_df_g2 is not None else pd.DataFrame()
    if len(sp): sp.to_csv(ART/f"{tag}_head_specialisation.csv", index=False); print(f"\n[Head Spec]\n{sp[['head_idx','dominant_view','entropy']].to_string(index=False)}")
    fig_dir = FIG/tag; fig_dir.mkdir(exist_ok=True)
    if pi_g2 is not None and len(sp): fig_head_affinity(sp, view_tags, fig_dir, tag); fig_pi_spread(pi_g2, view_tags, fig_dir, tag)
    if tau_g2 is not None: fig_tau(tau_g2, fig_dir, tag)
    fig_per_bin(bin_rows, task, fig_dir, tag)
    write_report(tag, task, metrics, sp, tau_g2 if tau_g2 is not None else [], view_tags, N_HEADS, REP/f"{tag}_report.md")
    print(f"\n[DONE] MNIST in {time.time()-t0:.1f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["california", "mnist", "both"], default="both")
    p.add_argument("--n-mnist", type=int, default=10000)
    a = p.parse_args()
    if a.dataset in ("california", "both"): run_california()
    if a.dataset in ("mnist", "both"):      run_mnist(a.n_mnist)


if __name__ == "__main__":
    main()
