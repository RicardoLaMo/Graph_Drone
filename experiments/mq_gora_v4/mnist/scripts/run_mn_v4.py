"""
run_mn_v4.py — MQ-GoRA v4 MNIST-784 (classification-enhanced track).

Variants:
  G2_ref    : v3 G2 carry-forward (reference)
  G10_ref   : v3 G10 exact reproduction (locked MNIST baseline)
  MN_v4a    : G10 exact reproduction under v4 training path
  MN_v4b    : MN_v4a + inter-head diversity regulariser (lam=0.005)
  MN_v4c    : MN_v4b + stronger diversity (lam=0.01)

Usage:
  cd /Volumes/MacMini/Projects/Graph_Drone
  python3 experiments/mq_gora_v4/mnist/scripts/run_mn_v4.py [--smoke] [--n_mnist N]
"""
import sys, os, time, argparse
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MN_DIR     = os.path.dirname(_SCRIPT_DIR)
_V4_DIR     = os.path.dirname(_MN_DIR)
_EXP_DIR    = os.path.dirname(_V4_DIR)
_REPO_ROOT  = os.path.dirname(_EXP_DIR)
_V3_SRC     = os.path.join(_EXP_DIR, 'gora_tabular', 'src')
_V4_SRC     = os.path.join(_V4_DIR, 'shared', 'src')

for p in [_V4_SRC, _V3_SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from observers import compute_observers
from views import mnist_view_features
from train import build_joint_neighbourhood, compute_label_ctx_per_view
from baselines import train_hgbr
from row_transformer import GoraTransformer

from manifold_teacher_v4 import ManifoldTeacher, train_teacher_v4
from row_transformer_v4 import MQGoraTransformerV4
from train_v4 import train_gora_v4_mn, predict_gora_v4
from eval_v4 import (
    compute_metrics_mn, compute_routing_stats,
    compute_regime_metrics, write_v4_report,
)

ART_DIR = os.path.join(_MN_DIR, 'artifacts')
REP_DIR = os.path.join(_MN_DIR, 'reports')
for d in [ART_DIR, REP_DIR]:
    os.makedirs(d, exist_ok=True)


def _make_mn_v4(tag, n_features, obs_dim, n_views, n_classes,
                use_label_ctx=True, use_teacher_query=True,
                use_alpha_gate=True, use_label_ctx_layernorm=False,
                d_model=64, n_heads=4):
    return MQGoraTransformerV4(
        n_features=n_features, obs_dim=obs_dim, n_views=n_views,
        out_dim=n_classes, d_model=d_model, n_heads=n_heads, n_layers=2,
        ff_dim=128, dropout=0.1, d_z=64, n_classes=n_classes,
        use_label_ctx=use_label_ctx,
        use_teacher_query=use_teacher_query,
        use_alpha_gate=use_alpha_gate,
        use_label_ctx_layernorm=use_label_ctx_layernorm,
    )


def _train_eval_mn(
    tag, model, X_all, g_all, y_all, neigh_idx, edge_wts,
    tr_i, va_i, te_i, view_mask, agree_score, kappa_all,
    view_tags, n_heads, n_classes,
    z_arr=None, lbl_nei=None,
    epochs=100, patience=20, lam_diversity=0.0,
):
    t0 = time.time()
    print(f"\n[{tag}] Training...")
    model, best_ep = train_gora_v4_mn(
        tag, model, X_all, g_all, y_all, neigh_idx, edge_wts,
        tr_i, va_i, task="classification",
        view_mask=view_mask, agree_score=agree_score,
        z_arr=z_arr, lbl_nei=lbl_nei,
        n_classes=n_classes, epochs=epochs, patience=patience,
        lam_diversity=lam_diversity,
    )

    preds, proba, pi_np, tau_np = predict_gora_v4(
        model, X_all, g_all, y_all, neigh_idx, edge_wts, te_i,
        task="classification", view_mask=view_mask, z_arr=z_arr,
    )
    metrics = compute_metrics_mn(y_all[te_i], preds, proba)
    routing = compute_routing_stats(pi_np, tau_np, view_tags, n_heads)
    regime  = compute_regime_metrics(y_all[te_i], preds, kappa_all, te_i, "classification")

    print(f"  [{tag}] Acc={metrics['accuracy']:.4f} F1={metrics['macro_f1']:.4f} "
          f"LogLoss={metrics['log_loss']:.4f}")

    routing.to_csv(os.path.join(ART_DIR, f"mn_v4_routing_{tag}.csv"), index=False)
    regime.to_csv(os.path.join(ART_DIR, f"mn_v4_regime_{tag}.csv"),   index=False)

    return {
        "tag": tag, "metrics": metrics, "routing": routing,
        "best_ep": best_ep, "stop_ep": "—", "collapsed": False,
    }


def run_mnist(n_subset=10000, smoke=False):
    print("=" * 60)
    print("  MQ-GoRA v4: MNIST-784 (classification-enhanced track)")
    print("=" * 60)

    if smoke:
        n_subset = 1000
        train_epochs, teacher_epochs = 5, 5
        print("[SMOKE] n=1000, 5 epochs")
    else:
        train_epochs, teacher_epochs = 100, 100

    print("[MN] Loading MNIST...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X_raw = mnist.data[:n_subset].astype(np.float32)
    y_raw = mnist.target[:n_subset].astype(int)
    N = len(X_raw)

    idx = np.arange(N)
    tr_va, te_i = train_test_split(idx, test_size=0.15, random_state=42)
    tr_i, va_i  = train_test_split(tr_va, test_size=0.177, random_state=42)

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_raw).astype(np.float32)
    print(f"[MN] N={N} train={len(tr_i)} val={len(va_i)} test={len(te_i)}")

    n_classes = 10
    view_feats, view_tags = mnist_view_features(X_all)
    n_views = len(view_tags)

    print("[MN] Computing observers...")
    g_all = compute_observers(X_all, view_feats, view_tags).astype(np.float32)
    kappa_all = g_all[:, 0]
    obs_dim = g_all.shape[1]

    print("[MN] Building joint neighbourhood...")
    neigh_idx, edge_wts, view_mask, agree_score = build_joint_neighbourhood(
        X_all, view_feats, view_tags, k_per_view=5)
    agree_mean = float(agree_score.mean())
    print(f"  [joint-kNN] mean agree_score={agree_mean:.3f}")

    n_features = X_all.shape[1]
    n_heads = 4

    # Teacher pre-training
    print("\n[MN] Teacher pre-training (full: L_agree+L_label+L_centroid)...")
    teacher = ManifoldTeacher(d_x=n_features, d_z=64, hidden=128)
    z_arr = train_teacher_v4(
        teacher, X_all, y_raw.astype(float), neigh_idx, edge_wts, view_mask,
        agree_score, tr_i,
        task="classification", n_classes=n_classes,
        epochs=teacher_epochs, skip_centroid_loss=False,
    )

    # Label context
    print("[MN] Precomputing label context...")
    lbl_nei = compute_label_ctx_per_view(y_raw.astype(float), neigh_idx, edge_wts, view_mask)

    results = []

    # ── B1_HGBR baseline ─────────────────────────────────────────────────────
    print("\n[B1] HGBR...")
    hgbr = train_hgbr(X_all, y_raw, tr_i, task="classification")
    preds_hgbr = hgbr.predict(X_all[te_i])
    m_hgbr = compute_metrics_mn(y_raw[te_i], preds_hgbr)
    print(f"  [B1_HGBR] Acc={m_hgbr['accuracy']:.4f}")
    results.append({"tag": "B1_HGBR", "metrics": m_hgbr, "routing": None,
                    "best_ep": "—", "stop_ep": "—", "collapsed": False})

    # ── G2_ref ───────────────────────────────────────────────────────────────
    print("\n[G2_ref] v3 GoRA carry-forward...")
    g2 = GoraTransformer(n_features=n_features, obs_dim=obs_dim, n_views=n_views,
                          out_dim=n_classes, d_model=64, n_heads=n_heads)
    from train import train_gora, predict_gora
    g2 = train_gora("G2_ref", g2, X_all, g_all, y_raw, neigh_idx, edge_wts,
                     tr_i, va_i, task="classification",
                     epochs=train_epochs if not smoke else 5)
    preds_g2, proba_g2, pi_g2, tau_g2 = predict_gora(
        g2, X_all, g_all, y_raw, neigh_idx, edge_wts, te_i, "classification")
    m_g2 = compute_metrics_mn(y_raw[te_i], preds_g2, proba_g2)
    r_g2 = compute_routing_stats(pi_g2, tau_g2, view_tags, n_heads)
    print(f"  [G2_ref] Acc={m_g2['accuracy']:.4f}")
    r_g2.to_csv(os.path.join(ART_DIR, "mn_v4_routing_G2_ref.csv"), index=False)
    results.append({"tag": "G2_ref", "metrics": m_g2, "routing": r_g2,
                    "best_ep": "—", "stop_ep": "—", "collapsed": False})

    # ── MN_v4a: G10 exact reproduction under v4 training path ────────────────
    m = _make_mn_v4("MN_v4a", n_features, obs_dim, n_views, n_classes,
                    use_label_ctx=True, use_teacher_query=True, use_alpha_gate=True)
    r = _train_eval_mn(
        "MN_v4a", m, X_all, g_all, y_raw, neigh_idx, edge_wts,
        tr_i, va_i, te_i, view_mask, agree_score, kappa_all, view_tags, n_heads, n_classes,
        z_arr=z_arr, lbl_nei=lbl_nei, epochs=train_epochs, patience=20,
    )
    results.append(r)

    # ── MN_v4b: MN_v4a + mild diversity regulariser ───────────────────────────
    m = _make_mn_v4("MN_v4b", n_features, obs_dim, n_views, n_classes,
                    use_label_ctx=True, use_teacher_query=True, use_alpha_gate=True)
    r = _train_eval_mn(
        "MN_v4b", m, X_all, g_all, y_raw, neigh_idx, edge_wts,
        tr_i, va_i, te_i, view_mask, agree_score, kappa_all, view_tags, n_heads, n_classes,
        z_arr=z_arr, lbl_nei=lbl_nei, epochs=train_epochs, patience=20,
        lam_diversity=0.005,
    )
    results.append(r)

    # ── MN_v4c: stronger diversity (lam=0.01) ─────────────────────────────────
    m = _make_mn_v4("MN_v4c", n_features, obs_dim, n_views, n_classes,
                    use_label_ctx=True, use_teacher_query=True, use_alpha_gate=True)
    r = _train_eval_mn(
        "MN_v4c", m, X_all, g_all, y_raw, neigh_idx, edge_wts,
        tr_i, va_i, te_i, view_mask, agree_score, kappa_all, view_tags, n_heads, n_classes,
        z_arr=z_arr, lbl_nei=lbl_nei, epochs=train_epochs, patience=20,
        lam_diversity=0.01,
    )
    results.append(r)

    # ── Write report ──────────────────────────────────────────────────────────
    import pandas as pd
    metrics_df = pd.DataFrame([
        {**{"tag": r["tag"]}, **r["metrics"]} for r in results
    ])
    metrics_df.to_csv(os.path.join(ART_DIR, "mn_v4_metrics.csv"), index=False)

    write_v4_report(
        dataset="mnist", task="classification",
        results=results,
        agree_score_mean=agree_mean,
        view_tags=view_tags,
        report_path=_MN_DIR + "/reports/mn_v4_report.md",
    )

    print(f"\n[DONE] MNIST v4")
    for r in results:
        m = r["metrics"]
        print(f"  {r['tag']:18s}  Acc={m.get('accuracy', float('nan')):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke",   action="store_true")
    ap.add_argument("--n_mnist", type=int, default=10000)
    args = ap.parse_args()
    run_mnist(n_subset=args.n_mnist, smoke=args.smoke)
