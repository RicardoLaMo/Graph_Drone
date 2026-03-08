"""
run_ca_v4.py — MQ-GoRA v4 California Housing (split-track, regression-safe).

Variants:
  G2_ref    : v3 G2 carry-forward (reference, no v4 changes)
  CA_v4a    : ViewSpecificEmbed + avg-pool ctx, NO label ctx         (CA_FIX_1 baseline)
  CA_v4b    : CA_v4a + normalised label ctx (y / y_std)             (CA_FIX_1)
  CA_v4c    : CA_v4b + LayerNorm on label_ctx_vec                    (CA_FIX_2)
  CA_v4d    : CA_v4c + teacher-lite (L_agree+L_label, 200ep)        (CA_FIX_3+4)
  CA_v4e    : CA_v4d + patience=40 + cosine annealing               (CA_FIX_5)

Usage:
  cd /Volumes/MacMini/Projects/Graph_Drone
  python3 experiments/mq_gora_v4/california/scripts/run_ca_v4.py [--smoke]
"""
import sys, os, time, argparse
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CA_DIR     = os.path.dirname(_SCRIPT_DIR)
_V4_DIR     = os.path.dirname(_CA_DIR)
_EXP_DIR    = os.path.dirname(_V4_DIR)
_REPO_ROOT  = os.path.dirname(_EXP_DIR)
_V3_SRC     = os.path.join(_EXP_DIR, 'gora_tabular', 'src')
_V4_SRC     = os.path.join(_V4_DIR, 'shared', 'src')

for p in [_V4_SRC, _V3_SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── v3 imports (unchanged components) ────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from observers import compute_observers
from views import california_view_features
from train import build_joint_neighbourhood, compute_label_ctx_per_view
from baselines import train_hgbr
from row_transformer import GoraTransformer               # G2 reference (v3)

# ── v4 imports ────────────────────────────────────────────────────────────────
from manifold_teacher_v4 import ManifoldTeacher, train_teacher_v4
from row_transformer_v4 import MQGoraTransformerV4
from train_v4 import (
    compute_y_norm_stats, normalise_lbl_nei,
    train_gora_v4_ca, predict_gora_v4,
)
from eval_v4 import (
    compute_metrics_ca, compute_routing_stats,
    compute_regime_metrics, write_v4_report,
)

# ── paths ─────────────────────────────────────────────────────────────────────
ART_DIR = os.path.join(_CA_DIR, 'artifacts')
REP_DIR = os.path.join(_CA_DIR, 'reports')
LOG_DIR = os.path.join(_CA_DIR, 'logs')
for d in [ART_DIR, REP_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _make_v4(tag, n_features, obs_dim, n_views, out_dim,
             use_label_ctx=False, use_teacher_query=False,
             use_alpha_gate=False, use_label_ctx_layernorm=False,
             n_classes=1, d_model=64, n_heads=4):
    return MQGoraTransformerV4(
        n_features=n_features, obs_dim=obs_dim, n_views=n_views, out_dim=out_dim,
        d_model=d_model, n_heads=n_heads, n_layers=2, ff_dim=128, dropout=0.1,
        d_z=64, n_classes=n_classes,
        use_label_ctx=use_label_ctx,
        use_teacher_query=use_teacher_query,
        use_alpha_gate=use_alpha_gate,
        use_label_ctx_layernorm=use_label_ctx_layernorm,
    )


def _train_eval_ca(
    tag, model, X_all, g_all, y_all, neigh_idx, edge_wts,
    tr_i, va_i, te_i, view_mask, agree_score, kappa_all,
    view_tags, n_heads,
    z_arr=None, lbl_nei=None,
    patience=20, use_cosine=False, epochs=150,
    change="", interp="",
):
    t0 = time.time()
    print(f"\n[{tag}] Training...")
    model = train_gora_v4_ca(
        tag, model, X_all, g_all, y_all, neigh_idx, edge_wts,
        tr_i, va_i,
        task="regression", view_mask=view_mask, agree_score=agree_score,
        z_arr=z_arr, lbl_nei=lbl_nei,
        epochs=epochs, patience=patience, use_cosine=use_cosine,
    )

    preds, _, pi_np, tau_np = predict_gora_v4(
        model, X_all, g_all, y_all, neigh_idx, edge_wts, te_i,
        task="regression", view_mask=view_mask, z_arr=z_arr,
    )
    metrics = compute_metrics_ca(y_all[te_i], preds)
    routing = compute_routing_stats(pi_np, tau_np, view_tags, n_heads)
    regime  = compute_regime_metrics(y_all[te_i], preds, kappa_all, te_i, "regression")

    print(f"  [{tag}] RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} R2={metrics['r2']:.4f}")

    # Save artifacts
    routing.to_csv(os.path.join(ART_DIR, f"ca_v4_routing_{tag}.csv"), index=False)
    regime.to_csv(os.path.join(ART_DIR, f"ca_v4_regime_{tag}.csv"),   index=False)

    return {
        "tag": tag, "metrics": metrics, "routing": routing,
        "best_ep": "—", "stop_ep": "—",
        "collapsed": False,
        "change": change, "interp": interp,
    }


# ─── main ────────────────────────────────────────────────────────────────────

def run_california(smoke=False):
    print("=" * 60)
    print("  MQ-GoRA v4: CALIFORNIA HOUSING (regression-safe track)")
    print("=" * 60)

    # Data
    data = fetch_california_housing()
    X_raw, y_raw = data.data.astype(np.float32), data.target.astype(np.float32)
    N = len(X_raw)

    if smoke:
        N = 800
        X_raw, y_raw = X_raw[:N], y_raw[:N]
        train_epochs, teacher_epochs = 5, 5
        print("[SMOKE] N=800, 5 epochs")
    else:
        train_epochs, teacher_epochs = 150, 200

    idx = np.arange(len(X_raw))
    tr_va, te = train_test_split(idx, test_size=0.15, random_state=42)
    tr_i, va_i = train_test_split(tr_va, test_size=0.177, random_state=42)

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_raw).astype(np.float32)

    print(f"[CA] N={len(X_raw)} train={len(tr_i)} val={len(va_i)} test={len(te)}")

    # Views and observers
    print("[CA] Computing views and observers...")
    view_feats, view_tags = california_view_features(X_all)
    n_views = len(view_tags)
    g_all = compute_observers(X_all, view_feats, view_tags).astype(np.float32)
    kappa_all = g_all[:, 0]  # first observer = kappa (curvature proxy)
    obs_dim = g_all.shape[1]

    # Joint neighbourhood
    print("[CA] Building joint neighbourhood...")
    neigh_idx, edge_wts, view_mask, agree_score = build_joint_neighbourhood(
        X_all, view_feats, view_tags, k_per_view=5)
    agree_mean = float(agree_score.mean())
    print(f"  [joint-kNN] mean agree_score={agree_mean:.3f}")

    n_features = X_all.shape[1]
    n_heads = 4
    out_dim = 1

    # Y normalisation stats (CA_FIX_1)
    y_mu, y_std = compute_y_norm_stats(y_raw, tr_i)
    print(f"[CA] y_norm: mu={y_mu:.3f} std={y_std:.3f}")

    # Precompute lbl_nei (raw)
    print("[CA] Precomputing label context (raw)...")
    lbl_nei_raw = compute_label_ctx_per_view(y_raw, neigh_idx, edge_wts, view_mask)

    # Normalised lbl_nei (CA_FIX_1)
    lbl_nei_norm = normalise_lbl_nei(lbl_nei_raw, y_mu, y_std)
    print(f"  lbl_nei_raw range: [{lbl_nei_raw.min():.2f}, {lbl_nei_raw.max():.2f}]")
    print(f"  lbl_nei_norm range: [{lbl_nei_norm.min():.2f}, {lbl_nei_norm.max():.2f}]")

    results = []

    # ── B1: HGBR baseline ────────────────────────────────────────────────────
    print("\n[B1] HGBR...")
    hgbr = train_hgbr(X_all, y_raw, tr_i, task="regression")
    preds_hgbr = hgbr.predict(X_all[te])
    m_hgbr = compute_metrics_ca(y_raw[te], preds_hgbr)
    print(f"  [B1_HGBR] RMSE={m_hgbr['rmse']:.4f} MAE={m_hgbr['mae']:.4f} R2={m_hgbr['r2']:.4f}")
    results.append({"tag": "B1_HGBR", "metrics": m_hgbr, "routing": None,
                    "best_ep": "—", "stop_ep": "—", "collapsed": False,
                    "change": "baseline", "interp": "Reference HGBR"})

    # ── G2_ref: v3 GoRA carry-forward ────────────────────────────────────────
    print("\n[G2_ref] v3 GoRA carry-forward...")
    g2 = GoraTransformer(n_features=n_features, obs_dim=obs_dim, n_views=n_views,
                          out_dim=out_dim, d_model=64, n_heads=n_heads)
    from train import train_gora, predict_gora
    g2 = train_gora("G2_ref", g2, X_all, g_all, y_raw, neigh_idx, edge_wts,
                     tr_i, va_i, task="regression",
                     epochs=train_epochs if not smoke else 5)
    preds_g2, _, pi_g2, tau_g2 = predict_gora(
        g2, X_all, g_all, y_raw, neigh_idx, edge_wts, te, "regression")
    m_g2 = compute_metrics_ca(y_raw[te], preds_g2)
    r_g2 = compute_routing_stats(pi_g2, tau_g2, view_tags, n_heads)
    print(f"  [G2_ref] RMSE={m_g2['rmse']:.4f} MAE={m_g2['mae']:.4f} R2={m_g2['r2']:.4f}")
    r_g2.to_csv(os.path.join(ART_DIR, "ca_v4_routing_G2_ref.csv"), index=False)
    results.append({"tag": "G2_ref", "metrics": m_g2, "routing": r_g2,
                    "best_ep": "—", "stop_ep": "—", "collapsed": False,
                    "change": "v3 reference", "interp": "GoRA v3 baseline"})

    # ── CA_v4a: no label ctx (ViewSpecific + avg-pool ctx, RichMoERouter) ────
    m = _make_v4("CA_v4a", n_features, obs_dim, n_views, out_dim)
    r = _train_eval_ca(
        "CA_v4a", m, X_all, g_all, y_raw, neigh_idx, edge_wts,
        tr_i, va_i, te, view_mask, agree_score, kappa_all, view_tags, n_heads,
        epochs=train_epochs,
        change="−LabelCtx  +ViewSpecificEmbed +ctx^(m)",
        interp="Removes label context entirely; tests if structure alone helps",
    )
    results.append(r)

    # ── CA_v4b: normalised label ctx ─────────────────────────────────────────
    m = _make_v4("CA_v4b", n_features, obs_dim, n_views, out_dim, use_label_ctx=True)
    r = _train_eval_ca(
        "CA_v4b", m, X_all, g_all, y_raw, neigh_idx, edge_wts,
        tr_i, va_i, te, view_mask, agree_score, kappa_all, view_tags, n_heads,
        lbl_nei=lbl_nei_norm, epochs=train_epochs,
        change="CA_v4a + y-normalised label ctx (÷y_std)",
        interp="CA_FIX_1: rescales label centroids to unit variance",
    )
    results.append(r)

    # ── CA_v4c: normalised label ctx + LayerNorm ──────────────────────────────
    m = _make_v4("CA_v4c", n_features, obs_dim, n_views, out_dim,
                 use_label_ctx=True, use_label_ctx_layernorm=True)
    r = _train_eval_ca(
        "CA_v4c", m, X_all, g_all, y_raw, neigh_idx, edge_wts,
        tr_i, va_i, te, view_mask, agree_score, kappa_all, view_tags, n_heads,
        lbl_nei=lbl_nei_norm, epochs=train_epochs,
        change="CA_v4b + LayerNorm(label_ctx_vec)",
        interp="CA_FIX_2: normalises the MLP output before router concatenation",
    )
    results.append(r)

    # ── Teacher-lite pre-training (used by CA_v4d and CA_v4e) ────────────────
    print("\n[CA] Teacher-lite pre-training (L_agree + L_label, 200 epochs)...")
    teacher = ManifoldTeacher(d_x=n_features, d_z=64, hidden=128)
    z_arr = train_teacher_v4(
        teacher, X_all, y_raw, neigh_idx, edge_wts, view_mask,
        agree_score, tr_i,
        task="regression", n_classes=1,
        epochs=teacher_epochs,
        skip_centroid_loss=True,   # CA_FIX_3
        lam_agree=1.0, lam_label=0.5,
    )

    # ── CA_v4d: + teacher-lite as cross-attn query ────────────────────────────
    m = _make_v4("CA_v4d", n_features, obs_dim, n_views, out_dim,
                 use_label_ctx=True, use_teacher_query=True,
                 use_label_ctx_layernorm=True)
    r = _train_eval_ca(
        "CA_v4d", m, X_all, g_all, y_raw, neigh_idx, edge_wts,
        tr_i, va_i, te, view_mask, agree_score, kappa_all, view_tags, n_heads,
        z_arr=z_arr, lbl_nei=lbl_nei_norm, epochs=train_epochs,
        change="CA_v4c + teacher-lite(L_agree+L_label, 200ep) as z_anc query",
        interp="CA_FIX_3+4: teacher without L_centroid, more epochs",
    )
    results.append(r)

    # ── CA_v4e: + patience=40 + cosine annealing ──────────────────────────────
    m = _make_v4("CA_v4e", n_features, obs_dim, n_views, out_dim,
                 use_label_ctx=True, use_teacher_query=True,
                 use_label_ctx_layernorm=True)
    r = _train_eval_ca(
        "CA_v4e", m, X_all, g_all, y_raw, neigh_idx, edge_wts,
        tr_i, va_i, te, view_mask, agree_score, kappa_all, view_tags, n_heads,
        z_arr=z_arr, lbl_nei=lbl_nei_norm,
        epochs=train_epochs, patience=40, use_cosine=True,
        change="CA_v4d + patience=40 + CosineAnnealing",
        interp="CA_FIX_5: healthier optimisation control",
    )
    results.append(r)

    # ── Write report ──────────────────────────────────────────────────────────
    import pandas as pd
    metrics_df = pd.DataFrame([
        {**{"tag": r["tag"]}, **r["metrics"]} for r in results
    ])
    metrics_df.to_csv(os.path.join(ART_DIR, "ca_v4_metrics.csv"), index=False)

    write_v4_report(
        dataset="california", task="regression",
        results=results,
        agree_score_mean=agree_mean,
        view_tags=view_tags,
        report_path=_CA_DIR + "/reports/ca_v4_report.md",
    )

    print(f"\n[DONE] California v4")
    for r in results:
        m = r["metrics"]
        tag = r["tag"]
        print(f"  {tag:18s}  RMSE={m.get('rmse', float('nan')):.4f}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run_california(smoke=args.smoke)
