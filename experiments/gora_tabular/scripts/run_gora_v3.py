"""
run_gora_v3.py — GoRA-Tabular v3: Manifold-Query GoRA (MQ-GoRA).

New models vs. v2:
  G7  — RichMoERouter: per-view neighbourhood ctx + ViewSpecificEmbedder
  G8  — G7 + LabelContextEncoder (router + value augmentation)
  G9  — G8 + ManifoldTeacher pre-training (z_anc as cross-attn query)
  G10 — G9 + AlphaGate prediction fusion (local-vs-transformer blend)

Ablation ladder:
  G8 - G7 = label dual-duty (routing + attention) in isolation
  G9 - G8 = teacher-as-query vs avg-pool (manifold-guided reading)
  G10 - G9 = alpha-gate prediction fusion in isolation

Baselines carried forward from v2: B1 (HGBR), B2 (TabPFN), G2 (GoRA v1).

Usage:
  python experiments/gora_tabular/scripts/run_gora_v3.py --dataset california
  python experiments/gora_tabular/scripts/run_gora_v3.py --dataset mnist
  python experiments/gora_tabular/scripts/run_gora_v3.py --dataset both
  python experiments/gora_tabular/scripts/run_gora_v3.py --dataset california --smoke
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
from experiments.gora_tabular.src.moe_router import MoERouter
from experiments.gora_tabular.src.row_transformer import GoraTransformer, MQGoraTransformer
from experiments.gora_tabular.src.manifold_teacher import ManifoldTeacher, train_teacher
from experiments.gora_tabular.src.baselines import train_hgbr, train_tabpfn
from experiments.gora_tabular.src.train import (
    build_neighbourhood, build_joint_neighbourhood,
    train_gora, predict_gora,
    compute_label_ctx_per_view,
    train_gora_v3, predict_gora_v3,
    get_device,
)
from experiments.gora_tabular.src.eval import (
    score, head_specialisation, bin_metric,
    fig_head_affinity, fig_pi_spread, fig_tau, fig_per_bin,
    write_report_v3,
)

SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED)

# Hyper-params matching v2 for fair comparison
D_MODEL = 64; N_HEADS = 4; N_LAYERS = 2; K_EACH = 5; K_SINGLE = 15; BATCH = 512
D_Z = 64; ROUTING_LAM = 0.05; LAM_ALPHA = 0.05

# Teacher
T_EPOCHS = 100; T_LR = 1e-3; T_LAM_AGREE = 1.0; T_LAM_LABEL = 0.5; T_LAM_CENTROID = 0.1


def _kappa_bins(kp):
    lo, hi = np.percentile(kp, 33), np.percentile(kp, 67)
    return np.where(kp <= lo, "low", np.where(kp <= hi, "medium", "high"))


def _make_mqgora(tag, n_features, obs_dim, n_views, out_dim,
                 use_label_ctx=False, use_teacher_query=False, use_alpha_gate=False,
                 n_classes=1):
    return MQGoraTransformer(
        n_features=n_features, obs_dim=obs_dim, n_views=n_views, out_dim=out_dim,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, ff_dim=128, dropout=0.1,
        d_z=D_Z, n_classes=n_classes,
        use_label_ctx=use_label_ctx,
        use_teacher_query=use_teacher_query,
        use_alpha_gate=use_alpha_gate,
        lam_alpha=LAM_ALPHA,
    )


def _train_eval_v3(tag, model, model_name, X, g, y, ni, ew, vm, ag, z_arr, lbl_nei,
                   tr_i, va_i, te_i, task, nc, kappa_bins, routing_lam=0.0):
    model = train_gora_v3(
        model, X, g, y, ni, ew, tr_i, va_i, task, nc,
        epochs=150, patience=20, lr=3e-4, batch_size=BATCH, name=model_name,
        view_mask=vm, agree_score=ag, routing_lam=routing_lam,
        z_arr=z_arr, lbl_nei=lbl_nei,
    )
    p, proba, pi_np = predict_gora_v3(
        model, X, g, y, ni, ew, te_i, task, view_mask=vm, z_arr=z_arr)
    m = score(model_name, y[te_i], p, task, proba)
    br = bin_metric(p, y[te_i], kappa_bins[te_i], task, model_name)
    if pi_np is not None:
        np.save(ART / f"{tag}_pi_{model_name}.npy", pi_np)
    return m, br, pi_np


# ─────────────────────────────────────────────────────────────────────────────
# California Housing
# ─────────────────────────────────────────────────────────────────────────────

def run_california(smoke=False):
    t0 = time.time(); tag = "california_v3"; task = "regression"; out_dim = 1; nc = 1
    print(f"\n{'='*60}\n  MQ-GoRA v3: CALIFORNIA HOUSING\n{'='*60}")

    cal = fetch_california_housing()
    X, y = cal.data.astype(np.float32), cal.target.astype(np.float32)
    X[:, [2, 4]] = np.log1p(X[:, [2, 4]])
    sc = RobustScaler(); X = sc.fit_transform(X).astype(np.float32)
    N = len(X)
    tr_i, tmp_i = train_test_split(np.arange(N), test_size=0.30, random_state=SEED)
    va_i, te_i = train_test_split(tmp_i, test_size=0.50, random_state=SEED)
    print(f"[CA] N={N} train={len(tr_i)} val={len(va_i)} test={len(te_i)}")

    if smoke:
        tr_i = tr_i[:500]; va_i = va_i[:200]; te_i = te_i[:200]
        t_epochs = 5
        print("[smoke] Truncated to 500/200/200 rows, 5 epochs")
    else:
        t_epochs = T_EPOCHS

    vfeats = california_view_features(X)
    view_tags = list(vfeats.keys()); M = len(view_tags)

    Xpca = PCA(n_components=4, random_state=SEED).fit_transform(X).astype(np.float32)
    print("[CA] Computing observers...")
    g, kp = compute_observers(Xpca, {k: vfeats[k] for k in ["GEO", "SOCIO", "LOWRANK"]}, k=K_SINGLE)
    obs_mean, obs_std = g[tr_i].mean(0), g[tr_i].std(0) + 1e-8
    g_sc = ((g - obs_mean) / obs_std).astype(np.float32)
    obs_dim = g_sc.shape[1]; kappa_bins = _kappa_bins(kp)

    # V2 joint neighbourhood (shared by all G7-G10)
    print("[CA] Building joint neighbourhood...")
    ni, ew, vm, ag = build_joint_neighbourhood(vfeats, k_per_view=K_EACH)
    P = ni.shape[1]; print(f"[CA] Pool P={P}")

    # V1 neighbourhood for G2 carry-forward
    ni_v1, ew_v1 = build_neighbourhood(vfeats, k=K_SINGLE, primary_key="GEO")

    # ── Teacher pre-training ──────────────────────────────────────────────────
    print("\n[Teacher] Pre-training ManifoldTeacher...")
    teacher = ManifoldTeacher(d_x=X.shape[1], d_z=D_Z, hidden=128)
    z_arr = train_teacher(
        teacher, X, y, ni, ew, vm, ag, tr_i,
        task=task, n_classes=nc,
        epochs=t_epochs, lr=T_LR,
        lam_agree=T_LAM_AGREE, lam_label=T_LAM_LABEL, lam_centroid=T_LAM_CENTROID,
    )

    # Label context for G8/G9/G10
    print("[CA] Precomputing label context per view...")
    lbl_nei = compute_label_ctx_per_view(y, ni, ew, vm)

    metrics, bin_rows, spec_rows = [], [], {}

    # ── B1 HGBR ───────────────────────────────────────────────────────────────
    print("[B1] HGBR..."); hgbr = train_hgbr(X[tr_i], y[tr_i], X[va_i], y[va_i], task)
    p_b1 = hgbr.predict(X[te_i]).astype(np.float32)
    metrics.append(score("B1_HGBR", y[te_i], p_b1, task))
    bin_rows += bin_metric(p_b1, y[te_i], kappa_bins[te_i], task, "B1_HGBR")

    # ── B2 TabPFN ─────────────────────────────────────────────────────────────
    if not smoke:
        print("[B2] TabPFN...")
        try:
            p_b2, _ = train_tabpfn(X[tr_i], y[tr_i], X[te_i], y[te_i], task,
                                    max_train_samples=8000, pca_features=False)
            metrics.append(score("B2_TabPFN", y[te_i], p_b2, task))
            bin_rows += bin_metric(p_b2, y[te_i], kappa_bins[te_i], task, "B2_TabPFN")
        except Exception as e:
            print(f"  [TabPFN] FAILED: {e}")
            metrics.append({"model": "B2_TabPFN", "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")})

    # ── G2 GoRA v1 (carry-forward reference) ─────────────────────────────────
    print("\n[G2] GoRA v1 carry-forward...")
    g2_model = GoraTransformer(X.shape[1], obs_dim, M, out_dim, D_MODEL, N_HEADS, N_LAYERS)
    g2_model = train_gora(g2_model, X, g_sc, y, ni_v1, ew_v1, tr_i, va_i, task, nc,
                          epochs=5 if smoke else 150, patience=20, lr=3e-4, batch_size=BATCH,
                          name="G2_GoRA_v1")
    p_g2, _, pi_g2, _ = predict_gora(g2_model, X, g_sc, y, ni_v1, ew_v1, te_i, task)
    metrics.append(score("G2_GoRA_v1", y[te_i], p_g2, task))
    bin_rows += bin_metric(p_g2, y[te_i], kappa_bins[te_i], task, "G2_GoRA_v1")
    if pi_g2 is not None: spec_rows["G2_GoRA_v1"] = head_specialisation(pi_g2, view_tags)

    # ── G7: RichCtx (ViewSpecificEmbed + avg-pool ctx, no label, no teacher) ──
    print("\n[G7] MQ-GoRA G7 (rich ctx, avg-pool)...")
    m, br, pi_g7 = _train_eval_v3(
        tag, _make_mqgora(tag, X.shape[1], obs_dim, M, out_dim, n_classes=nc),
        "G7_RichCtx", X, g_sc, y, ni, ew, vm, ag, None, None,
        tr_i, va_i, te_i, task, nc, kappa_bins)
    metrics.append(m); bin_rows += br
    if pi_g7 is not None: spec_rows["G7_RichCtx"] = head_specialisation(pi_g7, view_tags)

    # ── G8: + label context ───────────────────────────────────────────────────
    print("\n[G8] MQ-GoRA G8 (+label context)...")
    m, br, pi_g8 = _train_eval_v3(
        tag, _make_mqgora(tag, X.shape[1], obs_dim, M, out_dim, use_label_ctx=True, n_classes=nc),
        "G8_LabelCtx", X, g_sc, y, ni, ew, vm, ag, None, lbl_nei,
        tr_i, va_i, te_i, task, nc, kappa_bins)
    metrics.append(m); bin_rows += br
    if pi_g8 is not None: spec_rows["G8_LabelCtx"] = head_specialisation(pi_g8, view_tags)

    # ── G9: + teacher cross-attn query ───────────────────────────────────────
    print("\n[G9] MQ-GoRA G9 (+teacher query)...")
    m, br, pi_g9 = _train_eval_v3(
        tag, _make_mqgora(tag, X.shape[1], obs_dim, M, out_dim,
                          use_label_ctx=True, use_teacher_query=True, n_classes=nc),
        "G9_Teacher", X, g_sc, y, ni, ew, vm, ag, z_arr, lbl_nei,
        tr_i, va_i, te_i, task, nc, kappa_bins)
    metrics.append(m); bin_rows += br
    if pi_g9 is not None: spec_rows["G9_Teacher"] = head_specialisation(pi_g9, view_tags)

    # ── G10: Full MQ-GoRA ─────────────────────────────────────────────────────
    print("\n[G10] MQ-GoRA G10 (full: +alpha gate)...")
    m, br, pi_g10 = _train_eval_v3(
        tag, _make_mqgora(tag, X.shape[1], obs_dim, M, out_dim,
                          use_label_ctx=True, use_teacher_query=True, use_alpha_gate=True,
                          n_classes=nc),
        "G10_Full", X, g_sc, y, ni, ew, vm, ag, z_arr, lbl_nei,
        tr_i, va_i, te_i, task, nc, kappa_bins)
    metrics.append(m); bin_rows += br
    if pi_g10 is not None: spec_rows["G10_Full"] = head_specialisation(pi_g10, view_tags)

    # ── Outputs ───────────────────────────────────────────────────────────────
    pd.DataFrame(metrics).to_csv(ART / f"{tag}_metrics.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(ART / f"{tag}_bin_metrics.csv", index=False)
    fig_dir = FIG / tag; fig_dir.mkdir(exist_ok=True)
    for mname, sp in spec_rows.items():
        sp.to_csv(ART / f"{tag}_head_spec_{mname}.csv", index=False)
        if len(sp): fig_head_affinity(sp, view_tags, fig_dir, f"{tag}_{mname}")
    if pi_g10 is not None: fig_pi_spread(pi_g10, view_tags, fig_dir, f"{tag}_G10")
    fig_per_bin(bin_rows, task, fig_dir, tag)

    write_report_v3(tag, task, metrics, spec_rows, ag, view_tags, N_HEADS,
                    REP / f"{tag}_report.md")
    print(f"\n[DONE] California v3 in {time.time()-t0:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# MNIST-784
# ─────────────────────────────────────────────────────────────────────────────

def run_mnist(n_subset=10000, smoke=False):
    t0 = time.time(); tag = "mnist_v3"; task = "classification"; out_dim = 10; nc = 10
    print(f"\n{'='*60}\n  MQ-GoRA v3: MNIST-784\n{'='*60}")

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

    if smoke:
        tr_i = tr_i[:500]; va_i = va_i[:200]; te_i = te_i[:200]
        t_epochs = 5
        print("[smoke] Truncated to 500/200/200 rows, 5 epochs")
    else:
        t_epochs = T_EPOCHS

    vfeats = mnist_view_features(X)
    view_tags = list(vfeats.keys()); M = len(view_tags)

    Xpca50 = PCA(n_components=50, random_state=SEED).fit_transform(X).astype(np.float32)
    print("[MN] Computing observers...")
    g, kp = compute_observers(Xpca50, {k: vfeats[k] for k in ["BLOCK", "PCA"]}, k=K_SINGLE)
    obs_mean, obs_std = g[tr_i].mean(0), g[tr_i].std(0) + 1e-8
    g_sc = ((g - obs_mean) / obs_std).astype(np.float32)
    obs_dim = g_sc.shape[1]; kappa_bins = _kappa_bins(kp)

    print("[MN] Building joint neighbourhood...")
    ni, ew, vm, ag = build_joint_neighbourhood(vfeats, k_per_view=K_EACH)
    P = ni.shape[1]; print(f"[MN] Pool P={P}")

    ni_v1, ew_v1 = build_neighbourhood(vfeats, k=K_SINGLE, primary_key="PCA")

    # Teacher: classification — label context via class-index y (LabelContextEncoder handles it)
    print("\n[Teacher] Pre-training ManifoldTeacher (MNIST)...")
    teacher = ManifoldTeacher(d_x=X.shape[1], d_z=D_Z, hidden=128)
    # For clf, pass float(y) and n_classes so label centroid is per-view class mean
    z_arr = train_teacher(
        teacher, X, y.astype(np.float32), ni, ew, vm, ag, tr_i,
        task=task, n_classes=nc,
        epochs=t_epochs, lr=T_LR,
        lam_agree=T_LAM_AGREE, lam_label=T_LAM_LABEL, lam_centroid=T_LAM_CENTROID,
    )

    print("[MN] Precomputing label context per view...")
    lbl_nei = compute_label_ctx_per_view(y.astype(np.float32), ni, ew, vm)

    metrics, bin_rows, spec_rows = [], [], {}

    # B1
    print("[B1] HGBR..."); hgbr = train_hgbr(X[tr_i], y[tr_i], X[va_i], y[va_i], task)
    p_b1 = hgbr.predict(X[te_i]); pb1 = hgbr.predict_proba(X[te_i])
    metrics.append(score("B1_HGBR", y[te_i], p_b1, task, pb1))
    bin_rows += bin_metric(p_b1, y[te_i], kappa_bins[te_i], task, "B1_HGBR")

    # B2 TabPFN
    if not smoke:
        print("[B2] TabPFN...")
        try:
            p_b2, pb2 = train_tabpfn(X[tr_i], y[tr_i], X[te_i], y[te_i], task,
                                      max_train_samples=1000, max_features=200, pca_features=True)
            metrics.append(score("B2_TabPFN", y[te_i], p_b2, task, pb2))
            bin_rows += bin_metric(p_b2, y[te_i], kappa_bins[te_i], task, "B2_TabPFN")
        except Exception as e:
            print(f"  [TabPFN] FAILED: {e}")
            metrics.append({"model": "B2_TabPFN", "accuracy": float("nan"), "macro_f1": float("nan"), "log_loss": float("nan")})

    # G2 carry-forward
    print("\n[G2] GoRA v1 carry-forward...")
    g2_model = GoraTransformer(X.shape[1], obs_dim, M, out_dim, D_MODEL, N_HEADS, N_LAYERS)
    g2_model = train_gora(g2_model, X, g_sc, y, ni_v1, ew_v1, tr_i, va_i, task, nc,
                          epochs=5 if smoke else 150, patience=20, lr=3e-4, batch_size=BATCH,
                          name="G2_GoRA_v1")
    p_g2, pb_g2, pi_g2, _ = predict_gora(g2_model, X, g_sc, y, ni_v1, ew_v1, te_i, task)
    metrics.append(score("G2_GoRA_v1", y[te_i], p_g2, task, pb_g2))
    bin_rows += bin_metric(p_g2, y[te_i], kappa_bins[te_i], task, "G2_GoRA_v1")
    if pi_g2 is not None: spec_rows["G2_GoRA_v1"] = head_specialisation(pi_g2, view_tags)

    # G7-G10
    for model_tag, use_lbl, use_z, use_alpha, model_lbl_nei, model_z_arr in [
        ("G7_RichCtx",  False, False, False, None,     None),
        ("G8_LabelCtx", True,  False, False, lbl_nei,  None),
        ("G9_Teacher",  True,  True,  False, lbl_nei,  z_arr),
        ("G10_Full",    True,  True,  True,  lbl_nei,  z_arr),
    ]:
        print(f"\n[{model_tag}] MQ-GoRA...")
        m, br, pi = _train_eval_v3(
            tag,
            _make_mqgora(tag, X.shape[1], obs_dim, M, out_dim,
                         use_label_ctx=use_lbl, use_teacher_query=use_z,
                         use_alpha_gate=use_alpha, n_classes=nc),
            model_tag, X, g_sc, y, ni, ew, vm, ag, model_z_arr, model_lbl_nei,
            tr_i, va_i, te_i, task, nc, kappa_bins,
        )
        metrics.append(m); bin_rows += br
        if pi is not None: spec_rows[model_tag] = head_specialisation(pi, view_tags)

    # Outputs
    pd.DataFrame(metrics).to_csv(ART / f"{tag}_metrics.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(ART / f"{tag}_bin_metrics.csv", index=False)
    fig_dir = FIG / tag; fig_dir.mkdir(exist_ok=True)
    for mname, sp in spec_rows.items():
        sp.to_csv(ART / f"{tag}_head_spec_{mname}.csv", index=False)
        if len(sp): fig_head_affinity(sp, view_tags, fig_dir, f"{tag}_{mname}")
    fig_per_bin(bin_rows, task, fig_dir, tag)
    write_report_v3(tag, task, metrics, spec_rows, ag, view_tags, N_HEADS,
                    REP / f"{tag}_report.md")
    print(f"\n[DONE] MNIST v3 in {time.time()-t0:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["california", "mnist", "both"], default="both")
    p.add_argument("--n-mnist", type=int, default=10000)
    p.add_argument("--smoke", action="store_true",
                   help="Quick smoke test: 500 rows, 5 epochs per model")
    a = p.parse_args()
    if a.dataset in ("california", "both"): run_california(smoke=a.smoke)
    if a.dataset in ("mnist", "both"):      run_mnist(a.n_mnist, smoke=a.smoke)


if __name__ == "__main__":
    main()
