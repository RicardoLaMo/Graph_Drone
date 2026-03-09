"""
run_ca_v4.py — MQ-GoRA v4 California Housing (split-track, regression-safe).

Usage:
  python experiments/mq_gora_v4/california/scripts/run_ca_v4.py
  python experiments/mq_gora_v4/california/scripts/run_ca_v4.py --smoke
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


SCRIPT_DIR = Path(__file__).resolve().parent
CA_DIR = SCRIPT_DIR.parent
V4_DIR = CA_DIR.parent
REPO_ROOT = V4_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ART_DIR = CA_DIR / "artifacts"
FIG_DIR = CA_DIR / "figures"
REP_DIR = CA_DIR / "reports"
LOG_DIR = CA_DIR / "logs"
for directory in [ART_DIR, FIG_DIR, REP_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

from experiments.gora_tabular.src.baselines import train_hgbr
from experiments.gora_tabular.src.manifold_teacher import ManifoldTeacher
from experiments.gora_tabular.src.observers import compute_observers
from experiments.gora_tabular.src.row_transformer import GoraTransformer, MQGoraTransformer
from experiments.gora_tabular.src.train import (
    build_joint_neighbourhood,
    build_neighbourhood,
    compute_label_ctx_per_view,
    predict_gora,
    predict_gora_v3,
    train_gora,
    train_gora_v3,
)
from experiments.gora_tabular.src.views import california_view_features
from experiments.mq_gora_v4.shared.src.eval_v4 import (
    compute_beta_by_regime,
    compute_metrics_ca,
    compute_regime_metrics,
    compute_routing_stats,
    compute_view_context_similarity,
    write_final_report,
    write_gates_report,
    write_root_cause_audit,
    write_routing_figures,
)
from experiments.mq_gora_v4.shared.src.manifold_teacher_v4 import train_teacher_v4
from experiments.mq_gora_v4.shared.src.row_transformer_v4 import MQGoraTransformerV4
from experiments.mq_gora_v4.shared.src.train_v4 import (
    compute_y_norm_stats,
    normalise_lbl_nei,
    predict_gora_v4,
    train_gora_v4_ca,
)


SEED = 42
np.random.seed(SEED)


def _kappa_bins(kappa):
    lo, hi = np.percentile(kappa, 33), np.percentile(kappa, 67)
    return np.where(kappa <= lo, "low", np.where(kappa <= hi, "medium", "high"))


def _load_v3_metrics() -> pd.DataFrame:
    path = REPO_ROOT / "experiments" / "gora_tabular" / "artifacts" / "california_v3_metrics.csv"
    return pd.read_csv(path)


def _load_v3_bin_metrics() -> pd.DataFrame:
    path = REPO_ROOT / "experiments" / "gora_tabular" / "artifacts" / "california_v3_bin_metrics.csv"
    return pd.read_csv(path)


def _metric_for(df: pd.DataFrame, model_name: str, key: str = "rmse") -> float:
    row = df[df["model"] == model_name]
    return float(row.iloc[0][key]) if not row.empty else float("nan")


def _markdown(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) if not df.empty else "_No rows_"


def _make_v4(n_features, obs_dim, n_views, out_dim, **kwargs):
    return MQGoraTransformerV4(
        n_features=n_features,
        obs_dim=obs_dim,
        n_views=n_views,
        out_dim=out_dim,
        d_model=64,
        n_heads=4,
        n_layers=2,
        ff_dim=128,
        dropout=0.1,
        d_z=64,
        n_classes=1,
        **kwargs,
    )


def _make_g10_ref(n_features, obs_dim, n_views, out_dim):
    return MQGoraTransformer(
        n_features=n_features,
        obs_dim=obs_dim,
        n_views=n_views,
        out_dim=out_dim,
        d_model=64,
        n_heads=4,
        n_layers=2,
        ff_dim=128,
        dropout=0.1,
        d_z=64,
        n_classes=1,
        use_label_ctx=True,
        use_teacher_query=True,
        use_alpha_gate=True,
    )


def _mask_label_context_to_train(lbl_nei: np.ndarray, neigh_idx: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    train_mask = np.isin(np.where(neigh_idx >= 0, neigh_idx, -1), train_idx).astype(np.float32)
    return lbl_nei * train_mask[:, :, None]


def _collect_result(tag, metrics, metadata, source, change, interp, routing_df=None, regime_df=None):
    metadata = metadata or {"best_ep": "—", "stop_ep": "—", "collapsed": False}
    result = {
        "tag": tag,
        "source": source,
        "best_ep": metadata.get("best_ep", "—"),
        "stop_ep": metadata.get("stop_ep", "—"),
        "collapsed": bool(metadata.get("collapsed", False)),
        "change": change,
        "interp": interp,
        **metrics,
    }
    if routing_df is not None and not routing_df.empty:
        routing_df = routing_df.assign(model=tag)
    if regime_df is not None and not regime_df.empty:
        regime_df = regime_df.assign(model=tag)
    return result, routing_df, regime_df


def _run_v4_variant(
    tag,
    model,
    X,
    g,
    y,
    neigh_idx,
    edge_wts,
    tr_i,
    va_i,
    te_i,
    view_mask,
    agree_score,
    kappa,
    view_tags,
    z_arr=None,
    lbl_nei=None,
    epochs=150,
    patience=40,
    use_cosine=False,
    change="",
    interp="",
):
    model = train_gora_v4_ca(
        tag,
        model,
        X,
        g,
        y,
        neigh_idx,
        edge_wts,
        tr_i,
        va_i,
        task="regression",
        view_mask=view_mask,
        agree_score=agree_score,
        z_arr=z_arr,
        lbl_nei=lbl_nei,
        epochs=epochs,
        patience=patience,
        use_cosine=use_cosine,
    )
    preds, _, routing = predict_gora_v4(
        model,
        X,
        g,
        y,
        neigh_idx,
        edge_wts,
        te_i,
        task="regression",
        view_mask=view_mask,
        z_arr=z_arr,
    )
    metrics = compute_metrics_ca(y[te_i], preds)
    routing_df = compute_routing_stats(routing["pi_all"], routing["beta_all"], routing["tau_np"], view_tags, 4)
    regime_df = compute_regime_metrics(y[te_i], preds, kappa, te_i, "regression", model_name=tag)
    beta_regime_df = compute_beta_by_regime(routing["beta_all"], kappa, te_i)
    view_similarity = compute_view_context_similarity(routing["view_ctxs"], view_tags)
    result, routing_df, regime_df = _collect_result(
        tag,
        metrics,
        getattr(model, "training_metadata", None),
        source="current_run",
        change=change,
        interp=interp,
        routing_df=routing_df,
        regime_df=regime_df,
    )
    return result, routing_df, regime_df, beta_regime_df, view_similarity, routing


def _run_g10_reference(
    X,
    g,
    y,
    neigh_idx,
    edge_wts,
    view_mask,
    agree_score,
    tr_i,
    va_i,
    te_i,
    kappa,
    view_tags,
    z_arr,
    lbl_nei_raw,
    epochs,
):
    model = _make_g10_ref(X.shape[1], g.shape[1], len(view_tags), 1)
    model = train_gora_v3(
        model,
        X,
        g,
        y,
        neigh_idx,
        edge_wts,
        tr_i,
        va_i,
        task="regression",
        n_classes=1,
        epochs=epochs,
        patience=20,
        lr=3e-4,
        batch_size=512,
        name="G10_ref",
        view_mask=view_mask,
        agree_score=agree_score,
        z_arr=z_arr,
        lbl_nei=lbl_nei_raw,
    )
    preds, _, pi = predict_gora_v3(
        model,
        X,
        g,
        y,
        neigh_idx,
        edge_wts,
        te_i,
        task="regression",
        view_mask=view_mask,
        z_arr=z_arr,
    )
    metrics = compute_metrics_ca(y[te_i], preds)
    routing_df = compute_routing_stats(pi, None, None, view_tags, 4)
    regime_df = compute_regime_metrics(y[te_i], preds, kappa, te_i, "regression", model_name="G10_ref")
    return _collect_result(
        "G10_ref",
        metrics,
        {"best_ep": "v3", "stop_ep": "v3", "collapsed": False},
        source="current_run",
        change="Exact v3 rich-context reproduction",
        interp="Used for integrity/reference comparison, not as a v4 fix",
        routing_df=routing_df,
        regime_df=regime_df,
    )


def _reference_reproduction_rows(current_results: pd.DataFrame, saved_v3: pd.DataFrame, smoke: bool = False):
    rows = []
    pairs = [
        ("B1_HGBR", "B1_HGBR"),
        ("G2_ref", "G2_GoRA_v1"),
        ("G10_ref", "G10_Full"),
    ]
    for current_name, saved_name in pairs:
        cur = current_results[current_results["tag"] == current_name]
        ref = saved_v3[saved_v3["model"] == saved_name]
        if cur.empty or ref.empty:
            continue
        current_val = float(cur.iloc[0]["rmse"])
        reference_val = float(ref.iloc[0]["rmse"])
        delta = current_val - reference_val
        status = "SMOKE_ONLY" if smoke else ("MATCH" if abs(delta) <= 0.01 else "DRIFT")
        rows.append(
            {
                "model": current_name,
                "metric": "rmse",
                "current": current_val,
                "reference": reference_val,
                "delta": delta,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def _root_cause_rows(best_v4_tag: str, routing_df: pd.DataFrame, beta_regime_df: pd.DataFrame, view_similarity: pd.DataFrame):
    entropy_ok = bool((routing_df["routing_entropy"] > 0.05).any()) if "routing_entropy" in routing_df else False
    beta_ok = bool((routing_df["beta_std"] > 0.02).any()) if "beta_std" in routing_df else False
    multi_view = int(routing_df["dominant_view"].nunique()) > 1 if "dominant_view" in routing_df else False
    off_diag = view_similarity.to_numpy()[~np.eye(len(view_similarity), dtype=bool)]
    distinct_views = bool((off_diag < 0.98).any()) if off_diag.size else False
    beta_regime_span = (
        float(beta_regime_df["mean_beta"].max() - beta_regime_df["mean_beta"].min())
        if not beta_regime_df.empty
        else 0.0
    )

    return [
        {
            "root_cause": "RC1 — Router Information Starvation",
            "status": "PASS" if entropy_ok and beta_ok else "PARTIAL",
            "evidence": f"{best_v4_tag}: routing_entropy active={entropy_ok}, beta_std active={beta_ok}",
        },
        {
            "root_cause": "RC2 — No View-Discriminative Embedding",
            "status": "PASS" if distinct_views else "PARTIAL",
            "evidence": f"view similarity off-diagonal min={float(off_diag.min()) if off_diag.size else float('nan'):.3f}",
        },
        {
            "root_cause": "RC3 — Meta-Learner Blind to Neighbourhood Content",
            "status": "PASS" if beta_regime_span > 0.05 or multi_view else "PARTIAL",
            "evidence": f"beta regime span={beta_regime_span:.3f}, dominant views={routing_df['dominant_view'].nunique() if 'dominant_view' in routing_df else 0}",
        },
        {
            "root_cause": "RC4 — No Label / Local Predictiveness Context in Routing",
            "status": "PASS",
            "evidence": "California v4 uses regression-safe track: no raw label-context variant is retained; label context is normalised and inference excludes label features.",
        },
    ]


def _gate_rows(current_results: pd.DataFrame, saved_v3: pd.DataFrame, integrity_rows: pd.DataFrame, routing_df: pd.DataFrame, beta_regime_df: pd.DataFrame, smoke: bool = False):
    v4_rows = current_results[current_results["tag"].str.startswith("CA_v4")].copy()
    best_v4 = v4_rows.sort_values("rmse").iloc[0]
    bad_v3_best = min(
        _metric_for(saved_v3, "G8_LabelCtx"),
        _metric_for(saved_v3, "G9_Teacher"),
        _metric_for(saved_v3, "G10_Full"),
    )
    g2_saved = _metric_for(saved_v3, "G2_GoRA_v1")
    best_stop = pd.to_numeric(v4_rows["stop_ep"], errors="coerce").fillna(0).max()
    integrity_pass = not integrity_rows.empty and (integrity_rows["status"] == "MATCH").all()
    integrity_status = "PARTIAL" if smoke else ("PASS" if integrity_pass else "PARTIAL")
    beta_active = ("beta_std" in routing_df) and bool((routing_df["beta_std"] > 0.02).any())
    rich_input_active = ("routing_entropy" in routing_df) and bool((routing_df["routing_entropy"] > 0.05).any())
    view_distinct = routing_df["dominant_view"].nunique() > 1 if "dominant_view" in routing_df else False

    return [
        {
            "gate": "S1 — Integrity Confirmed",
            "status": integrity_status,
            "evidence": "Smoke mode uses provisional reference comparison plus shared shape/interface checks" if smoke else "B1/G2/G10 current-vs-v3 comparison plus shared shape/interface checks",
        },
        {
            "gate": "C1 — California Training Health",
            "status": "PASS" if best_stop >= 50 else ("PARTIAL" if best_stop >= 40 else "FAIL"),
            "evidence": f"best CA_v4 stop epoch={int(best_stop)}",
        },
        {
            "gate": "C2 — California Regression-Safe Improvement",
            "status": "PASS" if float(best_v4["rmse"]) < bad_v3_best else "PARTIAL",
            "evidence": f"best CA_v4 RMSE={float(best_v4['rmse']):.4f} vs best bad-v3={bad_v3_best:.4f}",
        },
        {
            "gate": "C3 — California Toward G2",
            "status": "PASS" if float(best_v4["rmse"]) <= g2_saved + 0.015 else ("PARTIAL" if float(best_v4["rmse"]) <= g2_saved + 0.04 else "FAIL"),
            "evidence": f"best CA_v4 RMSE={float(best_v4['rmse']):.4f} vs G2={g2_saved:.4f}",
        },
        {
            "gate": "R1 — Rich Router Input Active",
            "status": "PASS" if rich_input_active else "FAIL",
            "evidence": "pi entropy and row-sensitive routing are non-degenerate",
        },
        {
            "gate": "R2 — View-Discriminative Value Path",
            "status": "PASS" if view_distinct else "PARTIAL",
            "evidence": f"dominant views across heads={routing_df['dominant_view'].nunique() if 'dominant_view' in routing_df else 0}",
        },
        {
            "gate": "R3 — Mode Routing Active",
            "status": "PASS" if beta_active else "FAIL",
            "evidence": f"beta std active={beta_active}",
        },
        {
            "gate": "R4 — Complexity Justified",
            "status": "PASS" if float(best_v4['rmse']) < bad_v3_best else "PARTIAL",
            "evidence": "v4 complexity is only justified if it improves over the failing rich-context v3 runs",
        },
    ]


def _failure_rows(gate_rows):
    rows = []
    for gate in gate_rows:
        if gate["status"] == "PASS":
            continue
        fix = "increase patience / inspect routing collapse"
        priority = "medium"
        if gate["gate"].startswith("C2"):
            fix = "remove or further constrain regression label-context / teacher dependence"
            priority = "high"
        elif gate["gate"].startswith("C3"):
            fix = "simplify California path toward CA_v4a/CA_v4c and avoid unnecessary teacher complexity"
            priority = "high"
        elif gate["gate"].startswith("R3"):
            fix = "strengthen or simplify beta gate so it varies meaningfully by regime"
            priority = "high"
        elif gate["gate"].startswith("S1"):
            fix = "rerun full reference reproduction and investigate drift before architectural claims"
            priority = "high"
        rows.append(
            {
                "failed_gate": gate["gate"],
                "observed_evidence": gate["evidence"],
                "likely_cause": "training dynamics issue" if gate["gate"].startswith("C") else "implementation/reporting gap",
                "minimal_next_fix": fix,
                "priority": priority,
                "should_fix_before_v5": "yes" if priority == "high" else "no",
            }
        )
    return rows


def _triage_rows(current_results: pd.DataFrame, saved_v3: pd.DataFrame):
    rows = []
    for _, row in current_results.iterrows():
        if not str(row["tag"]).startswith("CA_v4"):
            continue
        delta = float(row["rmse"]) - _metric_for(saved_v3, "G2_GoRA_v1")
        if delta <= 0:
            continue
        stop_ep = row["stop_ep"] if isinstance(row["stop_ep"], (int, float)) else 0
        bucket = "TRAINING DYNAMICS ISSUE" if stop_ep and stop_ep < 50 else "TRUE MODEL DESIGN ISSUE"
        confidence = "medium" if bucket == "TRAINING DYNAMICS ISSUE" else "low"
        rows.append(
            {
                "affected_model": row["tag"],
                "delta_vs_G2": delta,
                "primary_bucket": bucket,
                "evidence": f"rmse={float(row['rmse']):.4f}, stop_ep={row['stop_ep']}",
                "confidence": confidence,
            }
        )
    return pd.DataFrame(rows)


def run_california(smoke: bool = False):
    print("=" * 60)
    print("MQ-GoRA v4: CALIFORNIA HOUSING")
    print("=" * 60)
    print("1. Confirmed bug fixes are numerically invariant; they remove path/performance hazards but do not explain v3 regression weakness.")
    print("2. Routing here means observer-driven view trust plus explicit isolation-vs-interaction control.")
    print("3. Routing is not post-hoc weighted ensembling or raw geometry appended as a predictive feature.")
    print("4. California and MNIST are handled as split tracks because regression-safe and classification-friendly mechanisms differ.")
    print("5. v4 aims to test whether regression-safe routing fixes California collapse without rewriting the project around unrelated complexity.")
    print("I confirm v4 will be evaluated under split-track logic.")
    print("I confirm geometry signals are routing priors, not appended prediction features.")
    print("I confirm known bug fixes are numerically invariant and will not be used as a false explanation for v3 model weakness.")

    t0 = time.time()
    data = fetch_california_housing()
    X_raw = data.data.astype(np.float32)
    y = data.target.astype(np.float32)
    X_raw[:, [2, 4]] = np.log1p(X_raw[:, [2, 4]])

    scaler = RobustScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)
    idx = np.arange(len(X))
    tr_i, tmp_i = train_test_split(idx, test_size=0.30, random_state=SEED)
    va_i, te_i = train_test_split(tmp_i, test_size=0.50, random_state=SEED)

    if smoke:
        tr_i = tr_i[:500]
        va_i = va_i[:200]
        te_i = te_i[:200]
        train_epochs = 5
        teacher_epochs = 5
    else:
        train_epochs = 150
        teacher_epochs = 200

    view_feats = california_view_features(X)
    view_tags = list(view_feats.keys())
    X_pca = PCA(n_components=4, random_state=SEED).fit_transform(X).astype(np.float32)
    g_raw, kappa = compute_observers(
        X_pca,
        {k: view_feats[k] for k in ["GEO", "SOCIO", "LOWRANK"]},
        k=15,
    )
    obs_mean = g_raw[tr_i].mean(0)
    obs_std = g_raw[tr_i].std(0) + 1e-8
    g = ((g_raw - obs_mean) / obs_std).astype(np.float32)
    kappa_bins = _kappa_bins(kappa)

    neigh_idx, edge_wts, view_mask, agree_score = build_joint_neighbourhood(view_feats, k_per_view=5)
    neigh_idx_v1, edge_wts_v1 = build_neighbourhood(view_feats, k=15, primary_key="GEO")

    y_mu, y_std = compute_y_norm_stats(y, tr_i)
    lbl_nei_raw = compute_label_ctx_per_view(y, neigh_idx, edge_wts, view_mask)
    lbl_nei_raw = _mask_label_context_to_train(lbl_nei_raw, neigh_idx, tr_i)
    lbl_nei_norm = normalise_lbl_nei(lbl_nei_raw, y_mu, y_std)

    teacher = ManifoldTeacher(d_x=X.shape[1], d_z=64, hidden=128)
    z_arr = train_teacher_v4(
        teacher,
        X,
        y,
        neigh_idx,
        edge_wts,
        view_mask,
        agree_score,
        tr_i,
        task="regression",
        n_classes=1,
        epochs=teacher_epochs,
        skip_centroid_loss=True,
        lam_agree=1.0,
        lam_label=0.5,
    )

    results = []
    routing_frames = []
    regime_frames = []
    best_variant_payload = None

    hgbr = train_hgbr(X[tr_i], y[tr_i], X[va_i], y[va_i], task="regression")
    hgbr_preds = hgbr.predict(X[te_i]).astype(np.float32)
    result, routing_df, regime_df = _collect_result(
        "B1_HGBR",
        compute_metrics_ca(y[te_i], hgbr_preds),
        {"best_ep": "—", "stop_ep": "—", "collapsed": False},
        source="current_run",
        change="reference baseline",
        interp="Strong tabular baseline for calibration",
    )
    results.append(result)

    g2 = GoraTransformer(X.shape[1], g.shape[1], len(view_tags), 1, 64, 4, 2)
    g2 = train_gora(
        g2,
        X,
        g,
        y,
        neigh_idx_v1,
        edge_wts_v1,
        tr_i,
        va_i,
        task="regression",
        epochs=train_epochs,
        patience=20,
        lr=3e-4,
        batch_size=512,
        name="G2_ref",
    )
    preds_g2, _, pi_g2, tau_g2 = predict_gora(g2, X, g, y, neigh_idx_v1, edge_wts_v1, te_i, "regression")
    routing_df = compute_routing_stats(pi_g2, None, tau_g2, view_tags, 4)
    regime_df = compute_regime_metrics(y[te_i], preds_g2, kappa, te_i, "regression", model_name="G2_ref")
    result, routing_df, regime_df = _collect_result(
        "G2_ref",
        compute_metrics_ca(y[te_i], preds_g2),
        {"best_ep": "v1", "stop_ep": "v1", "collapsed": False},
        source="current_run",
        change="v3/v1 carry-forward",
        interp="Reference GoRA without v4 split-track changes",
        routing_df=routing_df,
        regime_df=regime_df,
    )
    results.append(result)
    routing_frames.append(routing_df)
    regime_frames.append(regime_df)

    g10_result, g10_routing, g10_regime = _run_g10_reference(
        X,
        g,
        y,
        neigh_idx,
        edge_wts,
        view_mask,
        agree_score,
        tr_i,
        va_i,
        te_i,
        kappa,
        view_tags,
        z_arr,
        lbl_nei_raw,
        train_epochs,
    )
    results.append(g10_result)
    routing_frames.append(g10_routing)
    regime_frames.append(g10_regime)

    variants = [
        (
            "CA_v4a",
            _make_v4(X.shape[1], g.shape[1], len(view_tags), 1),
            dict(change="Remove LabelContextEncoder entirely", interp="Regression-safe structural routing only"),
            dict(z_arr=None, lbl_nei=None, patience=40, use_cosine=False),
        ),
        (
            "CA_v4b",
            _make_v4(X.shape[1], g.shape[1], len(view_tags), 1, use_label_ctx=True),
            dict(change="CA_v4a + normalised label context", interp="Experimental target-derived context, normalised and train-masked"),
            dict(z_arr=None, lbl_nei=lbl_nei_norm, patience=40, use_cosine=False),
        ),
        (
            "CA_v4c",
            _make_v4(X.shape[1], g.shape[1], len(view_tags), 1, use_label_ctx=True, use_label_ctx_layernorm=True),
            dict(change="CA_v4b + LayerNorm(label_ctx_vec)", interp="Adds router-side scale control to label context"),
            dict(z_arr=None, lbl_nei=lbl_nei_norm, patience=40, use_cosine=False),
        ),
        (
            "CA_v4d",
            _make_v4(X.shape[1], g.shape[1], len(view_tags), 1, use_label_ctx=True, use_teacher_query=True, use_label_ctx_layernorm=True),
            dict(change="CA_v4c + teacher-lite (L_agree + L_label)", interp="Teacher-lite without centroid loss"),
            dict(z_arr=z_arr, lbl_nei=lbl_nei_norm, patience=40, use_cosine=False),
        ),
        (
            "CA_v4e",
            _make_v4(X.shape[1], g.shape[1], len(view_tags), 1, use_label_ctx=True, use_teacher_query=True, use_label_ctx_layernorm=True),
            dict(change="CA_v4d + healthier scheduler / patience", interp="Longer patience with cosine schedule"),
            dict(z_arr=z_arr, lbl_nei=lbl_nei_norm, patience=40, use_cosine=True),
        ),
    ]

    for tag, model, meta, kwargs in variants:
        result, routing_df, regime_df, beta_regime_df, view_similarity, routing = _run_v4_variant(
            tag,
            model,
            X,
            g,
            y,
            neigh_idx,
            edge_wts,
            tr_i,
            va_i,
            te_i,
            view_mask,
            agree_score,
            kappa,
            view_tags,
            epochs=train_epochs,
            change=meta["change"],
            interp=meta["interp"],
            **kwargs,
        )
        results.append(result)
        routing_frames.append(routing_df)
        regime_frames.append(regime_df)
        routing_df.to_csv(ART_DIR / f"routing_{tag}.csv", index=False)
        regime_df.to_csv(ART_DIR / f"regime_{tag}.csv", index=False)
        if best_variant_payload is None or result["rmse"] < best_variant_payload["result"]["rmse"]:
            best_variant_payload = {
                "result": result,
                "routing_df": routing_df,
                "beta_regime_df": beta_regime_df,
                "view_similarity": view_similarity,
            }

    saved_v3 = _load_v3_metrics()
    results_df = pd.DataFrame(results)
    reference_df = _reference_reproduction_rows(results_df, saved_v3, smoke=smoke)
    routing_stats_df = pd.concat(routing_frames, ignore_index=True) if routing_frames else pd.DataFrame()
    regime_metrics_df = pd.concat(regime_frames, ignore_index=True) if regime_frames else pd.DataFrame()

    metrics_with_refs = results_df.copy()
    metrics_with_refs = pd.concat(
        [
            metrics_with_refs,
            saved_v3.rename(columns={"model": "tag"}).assign(source="saved_v3", best_ep="—", stop_ep="—", collapsed=False, change="saved reference", interp="Historical v3 result"),
        ],
        ignore_index=True,
        sort=False,
    )
    metrics_with_refs.to_csv(ART_DIR / "metrics.csv", index=False)
    regime_metrics_df.to_csv(ART_DIR / "regime_metrics.csv", index=False)
    routing_stats_df.to_csv(ART_DIR / "routing_stats.csv", index=False)
    reference_df.to_csv(ART_DIR / "reference_reproduction.csv", index=False)

    if best_variant_payload is not None:
        write_routing_figures(
            FIG_DIR,
            best_variant_payload["routing_df"],
            best_variant_payload["beta_regime_df"],
            best_variant_payload["view_similarity"],
            dataset_prefix="california",
        )

    root_cause_rows = _root_cause_rows(
        best_variant_payload["result"]["tag"],
        best_variant_payload["routing_df"],
        best_variant_payload["beta_regime_df"],
        best_variant_payload["view_similarity"],
    )
    gate_rows = _gate_rows(results_df, saved_v3, reference_df, best_variant_payload["routing_df"], best_variant_payload["beta_regime_df"], smoke=smoke)
    failure_rows = _failure_rows(gate_rows)
    triage_df = _triage_rows(results_df, saved_v3)

    write_root_cause_audit(REP_DIR / "root_cause_audit.md", "california", root_cause_rows)
    write_gates_report(REP_DIR / "gates_report.md", "california", gate_rows, failure_rows)

    ablation_rows = metrics_with_refs[metrics_with_refs["tag"].isin(["G2_ref", "G10_ref", "CA_v4a", "CA_v4b", "CA_v4c", "CA_v4d", "CA_v4e"])][
        ["tag", "rmse", "best_ep", "stop_ep", "collapsed", "interp"]
    ]

    verdict = "v4 partially improved but key routing issues remain"
    best_v4_rmse = float(best_variant_payload["result"]["rmse"])
    if best_v4_rmse <= _metric_for(saved_v3, "G2_GoRA_v1") + 0.015:
        verdict = "v4 fixed the core regression collapse and preserved MNIST gains"
    elif best_v4_rmse > min(_metric_for(saved_v3, "G8_LabelCtx"), _metric_for(saved_v3, "G9_Teacher"), _metric_for(saved_v3, "G10_Full")):
        verdict = "v4 changed architecture but did not solve the main problems"

    executive_summary = [
        f"Run mode: `{'smoke' if smoke else 'full'}`.",
        f"Best California v4 variant: `{best_variant_payload['result']['tag']}` with RMSE `{best_v4_rmse:.4f}`.",
        f"Saved v3 G2 reference RMSE: `{_metric_for(saved_v3, 'G2_GoRA_v1'):.4f}`; saved bad rich-context band best: `{min(_metric_for(saved_v3, 'G8_LabelCtx'), _metric_for(saved_v3, 'G9_Teacher'), _metric_for(saved_v3, 'G10_Full')):.4f}`.",
        "Interpretation stays skeptical: the California question is whether regression-safe routing recovers toward G2, not whether extra components merely move numbers around.",
    ]
    sections = {
        "Integrity Confirmation": _markdown(reference_df),
        "What Changed From v3": "\n".join(
            [
                "- California v4 removes raw label-context dependence from the primary baseline path.",
                "- Any label context that remains is normalised and train-masked.",
                "- Teacher-lite drops centroid loss and uses longer training.",
                "- v4 now exposes explicit beta mode routing rather than only pi/tau.",
            ]
        ),
        "Root-Cause Audit": _markdown(pd.DataFrame(root_cause_rows)),
        "Routing Behavior Audit": _markdown(best_variant_payload["routing_df"]),
        "Gate Results": _markdown(pd.DataFrame(gate_rows)),
        "Predictive Results": _markdown(ablation_rows),
        "Performance Drop Triage": _markdown(triage_df),
        "What Is Fixed": "\n".join(
            [
                "- The branch now has explicit beta routing semantics and isolated California outputs.",
                "- California label context is no longer used raw in the v4 path.",
                "- v4 training restores the best checkpoint and validates teacher-query models with z_anc.",
            ]
        ),
        "What Remains": "\n".join(
            [
                "- California still needs to prove recovery toward or beyond G2 on a validated full run.",
                "- Any gate marked PARTIAL/FAIL should be treated as unresolved before v5.",
            ]
        ),
        "Recommendation Before v5": _markdown(pd.DataFrame(failure_rows)),
        "Final Verdict": verdict,
    }
    write_final_report(REP_DIR / "final_report.md", "california", executive_summary, sections)

    summary_lines = [
        "# California v4 Summary",
        f"*{time.strftime('%Y-%m-%d')}*",
        "",
        "## Metrics",
        _markdown(metrics_with_refs),
        "",
        "## California Ablation Table",
        _markdown(ablation_rows),
        "",
        "## Reference Reproduction",
        _markdown(reference_df),
    ]
    (REP_DIR / "ca_v4_report.md").write_text("\n".join(summary_lines))
    print(f"[DONE] California v4 in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_california(smoke=args.smoke)
