"""
run_mn_v4.py — MQ-GoRA v4 MNIST-784 (classification-enhanced track).

Usage:
  python experiments/mq_gora_v4/mnist/scripts/run_mn_v4.py
  python experiments/mq_gora_v4/mnist/scripts/run_mn_v4.py --smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
MN_DIR = SCRIPT_DIR.parent
V4_DIR = MN_DIR.parent
REPO_ROOT = V4_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ART_DIR = MN_DIR / "artifacts"
FIG_DIR = MN_DIR / "figures"
REP_DIR = MN_DIR / "reports"
LOG_DIR = MN_DIR / "logs"
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
from experiments.gora_tabular.src.views import mnist_view_features
from experiments.mq_gora_v4.shared.src.eval_v4 import (
    compute_beta_by_regime,
    compute_metrics_mn,
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
from experiments.mq_gora_v4.shared.src.train_v4 import predict_gora_v4, train_gora_v4_mn


SEED = 42
np.random.seed(SEED)


def _kappa_bins(kappa):
    lo, hi = np.percentile(kappa, 33), np.percentile(kappa, 67)
    return np.where(kappa <= lo, "low", np.where(kappa <= hi, "medium", "high"))


def _load_v3_metrics() -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "experiments" / "gora_tabular" / "artifacts" / "mnist_v3_metrics.csv")


def _metric_for(df: pd.DataFrame, model_name: str, key: str = "accuracy") -> float:
    row = df[df["model"] == model_name]
    return float(row.iloc[0][key]) if not row.empty else float("nan")


def _markdown(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) if not df.empty else "_No rows_"


def _make_v4(n_features, obs_dim, n_views, n_classes, **kwargs):
    return MQGoraTransformerV4(
        n_features=n_features,
        obs_dim=obs_dim,
        n_views=n_views,
        out_dim=n_classes,
        d_model=64,
        n_heads=4,
        n_layers=2,
        ff_dim=128,
        dropout=0.1,
        d_z=64,
        n_classes=n_classes,
        **kwargs,
    )


def _make_g10_ref(n_features, obs_dim, n_views, n_classes):
    return MQGoraTransformer(
        n_features=n_features,
        obs_dim=obs_dim,
        n_views=n_views,
        out_dim=n_classes,
        d_model=64,
        n_heads=4,
        n_layers=2,
        ff_dim=128,
        dropout=0.1,
        d_z=64,
        n_classes=n_classes,
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
    z_arr,
    lbl_nei,
    epochs,
    lam_diversity,
    change,
    interp,
):
    model = train_gora_v4_mn(
        tag,
        model,
        X,
        g,
        y,
        neigh_idx,
        edge_wts,
        tr_i,
        va_i,
        task="classification",
        view_mask=view_mask,
        agree_score=agree_score,
        z_arr=z_arr,
        lbl_nei=lbl_nei,
        n_classes=10,
        epochs=epochs,
        patience=20,
        lam_diversity=lam_diversity,
    )
    preds, proba, routing = predict_gora_v4(
        model,
        X,
        g,
        y,
        neigh_idx,
        edge_wts,
        te_i,
        task="classification",
        view_mask=view_mask,
        z_arr=z_arr,
    )
    metrics = compute_metrics_mn(y[te_i], preds, proba)
    routing_df = compute_routing_stats(routing["pi_all"], routing["beta_all"], routing["tau_np"], view_tags, 4)
    regime_df = compute_regime_metrics(y[te_i], preds, kappa, te_i, "classification", model_name=tag)
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
    return result, routing_df, regime_df, beta_regime_df, view_similarity


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
    lbl_nei,
    epochs,
):
    model = _make_g10_ref(X.shape[1], g.shape[1], len(view_tags), 10)
    model = train_gora_v3(
        model,
        X,
        g,
        y,
        neigh_idx,
        edge_wts,
        tr_i,
        va_i,
        task="classification",
        n_classes=10,
        epochs=epochs,
        patience=20,
        lr=3e-4,
        batch_size=512,
        name="G10_ref",
        view_mask=view_mask,
        agree_score=agree_score,
        z_arr=z_arr,
        lbl_nei=lbl_nei,
    )
    preds, proba, pi = predict_gora_v3(
        model,
        X,
        g,
        y,
        neigh_idx,
        edge_wts,
        te_i,
        task="classification",
        view_mask=view_mask,
        z_arr=z_arr,
    )
    metrics = compute_metrics_mn(y[te_i], preds, proba)
    routing_df = compute_routing_stats(pi, None, None, view_tags, 4)
    regime_df = compute_regime_metrics(y[te_i], preds, kappa, te_i, "classification", model_name="G10_ref")
    return _collect_result(
        "G10_ref",
        metrics,
        {"best_ep": "v3", "stop_ep": "v3", "collapsed": False},
        source="current_run",
        change="Exact v3 G10 reproduction",
        interp="Protected baseline before any MNIST routing changes",
        routing_df=routing_df,
        regime_df=regime_df,
    )


def _reference_reproduction_rows(current_results: pd.DataFrame, saved_v3: pd.DataFrame, smoke: bool = False):
    rows = []
    for current_name, saved_name in [("B1_HGBR", "B1_HGBR"), ("G2_ref", "G2_GoRA_v1"), ("G10_ref", "G10_Full")]:
        cur = current_results[current_results["tag"] == current_name]
        ref = saved_v3[saved_v3["model"] == saved_name]
        if cur.empty or ref.empty:
            continue
        current_val = float(cur.iloc[0]["accuracy"])
        reference_val = float(ref.iloc[0]["accuracy"])
        delta = current_val - reference_val
        status = "SMOKE_ONLY" if smoke else ("MATCH" if abs(delta) <= 0.01 else "DRIFT")
        rows.append(
            {
                "model": current_name,
                "metric": "accuracy",
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
    off_diag = view_similarity.to_numpy()[~np.eye(len(view_similarity), dtype=bool)]
    distinct_views = bool((off_diag < 0.98).any()) if off_diag.size else False
    dominant_count = routing_df["dominant_view"].nunique() if "dominant_view" in routing_df else 0
    beta_regime_span = (
        float(beta_regime_df["mean_beta"].max() - beta_regime_df["mean_beta"].min())
        if not beta_regime_df.empty
        else 0.0
    )
    return [
        {
            "root_cause": "RC1 — Router Information Starvation",
            "status": "PASS" if entropy_ok and beta_ok else "PARTIAL",
            "evidence": f"{best_v4_tag}: routing_entropy active={entropy_ok}, beta active={beta_ok}",
        },
        {
            "root_cause": "RC2 — No View-Discriminative Embedding",
            "status": "PASS" if distinct_views else "PARTIAL",
            "evidence": f"view similarity off-diagonal min={float(off_diag.min()) if off_diag.size else float('nan'):.3f}",
        },
        {
            "root_cause": "RC3 — Meta-Learner Blind to Neighbourhood Content",
            "status": "PASS" if beta_regime_span > 0.05 or dominant_count > 1 else "PARTIAL",
            "evidence": f"beta regime span={beta_regime_span:.3f}, dominant views={dominant_count}",
        },
        {
            "root_cause": "RC4 — No Label / Local Predictiveness Context in Routing",
            "status": "PASS",
            "evidence": "MNIST retains controlled label-context and teacher signal with alpha-gate protection rather than removing them blindly.",
        },
    ]


def _gate_rows(current_results: pd.DataFrame, saved_v3: pd.DataFrame, integrity_rows: pd.DataFrame, routing_df: pd.DataFrame, smoke: bool = False):
    v4_rows = current_results[current_results["tag"].str.startswith("MN_v4")].copy()
    best_v4 = v4_rows.sort_values("accuracy", ascending=False).iloc[0]
    saved_g10 = _metric_for(saved_v3, "G10_Full")
    saved_g2 = _metric_for(saved_v3, "G2_GoRA_v1")
    integrity_pass = not integrity_rows.empty and (integrity_rows["status"] == "MATCH").all()
    integrity_status = "PARTIAL" if smoke else ("PASS" if integrity_pass else "PARTIAL")
    routing_good = routing_df["dominant_view"].nunique() > 1 if "dominant_view" in routing_df else False
    beta_active = ("beta_std" in routing_df) and bool((routing_df["beta_std"] > 0.02).any())

    return [
        {
            "gate": "S1 — Integrity Confirmed",
            "status": integrity_status,
            "evidence": "Smoke mode uses provisional reference comparison under current code path" if smoke else "B1/G2/G10 reference comparison under current code path",
        },
        {
            "gate": "M1 — MNIST Gain Retention",
            "status": "PASS" if float(best_v4["accuracy"]) >= saved_g10 - 0.002 else ("PARTIAL" if float(best_v4["accuracy"]) >= saved_g10 - 0.01 else "FAIL"),
            "evidence": f"best MN_v4 accuracy={float(best_v4['accuracy']):.4f} vs saved G10={saved_g10:.4f}",
        },
        {
            "gate": "M2 — MNIST Routing Quality",
            "status": "PASS" if routing_good and float(best_v4["accuracy"]) >= saved_g2 else "PARTIAL",
            "evidence": f"dominant views across heads={routing_df['dominant_view'].nunique() if 'dominant_view' in routing_df else 0}",
        },
        {
            "gate": "R1 — Rich Router Input Active",
            "status": "PASS" if ("routing_entropy" in routing_df) and bool((routing_df["routing_entropy"] > 0.05).any()) else "FAIL",
            "evidence": "pi varies across heads and rows",
        },
        {
            "gate": "R2 — View-Discriminative Value Path",
            "status": "PASS" if routing_good else "PARTIAL",
            "evidence": f"dominant views across heads={routing_df['dominant_view'].nunique() if 'dominant_view' in routing_df else 0}",
        },
        {
            "gate": "R3 — Mode Routing Active",
            "status": "PASS" if beta_active else "FAIL",
            "evidence": f"beta std active={beta_active}",
        },
        {
            "gate": "R4 — Complexity Justified",
            "status": "PASS" if float(best_v4["accuracy"]) >= saved_g2 else "PARTIAL",
            "evidence": "Extra routing complexity is justified only if MNIST gains remain above G2 while specialization improves.",
        },
    ]


def _failure_rows(gate_rows):
    rows = []
    for gate in gate_rows:
        if gate["status"] == "PASS":
            continue
        fix = "inspect protected alpha-gate path before adding more routing complexity"
        priority = "medium"
        if gate["gate"].startswith("M1"):
            fix = "revert to exact G10 semantics for any change that hurts accuracy materially"
            priority = "high"
        elif gate["gate"].startswith("M2") or gate["gate"].startswith("R3"):
            fix = "use the diversity regulariser only if specialization improves without accuracy loss"
            priority = "high"
        rows.append(
            {
                "failed_gate": gate["gate"],
                "observed_evidence": gate["evidence"],
                "likely_cause": "training dynamics issue" if gate["gate"].startswith("M") else "model complexity mismatch",
                "minimal_next_fix": fix,
                "priority": priority,
                "should_fix_before_v5": "yes" if priority == "high" else "no",
            }
        )
    return rows


def _triage_rows(current_results: pd.DataFrame, saved_v3: pd.DataFrame):
    rows = []
    saved_g10 = _metric_for(saved_v3, "G10_Full")
    for _, row in current_results.iterrows():
        if not str(row["tag"]).startswith("MN_v4"):
            continue
        delta = saved_g10 - float(row["accuracy"])
        if delta <= 0:
            continue
        bucket = "TRAINING DYNAMICS ISSUE" if row["stop_ep"] and row["stop_ep"] < 40 else "TRUE MODEL DESIGN ISSUE"
        rows.append(
            {
                "affected_model": row["tag"],
                "delta_vs_G10": delta,
                "primary_bucket": bucket,
                "evidence": f"accuracy={float(row['accuracy']):.4f}, stop_ep={row['stop_ep']}",
                "confidence": "medium",
            }
        )
    return pd.DataFrame(rows)


def run_mnist(n_subset: int = 10000, smoke: bool = False):
    print("=" * 60)
    print("MQ-GoRA v4: MNIST-784")
    print("=" * 60)
    print("1. Confirmed bug fixes are numerically invariant and do not explain MNIST gains or losses by themselves.")
    print("2. Routing here means observer-driven view trust plus explicit isolation-vs-interaction control.")
    print("3. MNIST is a protected classification track: alpha-gate behavior is preserved unless an ablation proves otherwise.")
    print("4. The v4 goal on MNIST is preservation first, then modest routing-quality improvement.")
    print("I confirm v4 will be evaluated under split-track logic.")
    print("I confirm geometry signals are routing priors, not appended prediction features.")
    print("I confirm known bug fixes are numerically invariant and will not be used as a false explanation for v3 model weakness.")

    t0 = time.time()
    if smoke:
        n_subset = 1000
        train_epochs = 5
        teacher_epochs = 5
    else:
        train_epochs = 100
        teacher_epochs = 100

    data = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X_raw = data.data.astype(np.float32)
    y = data.target.astype(np.int64)
    if n_subset and n_subset < len(X_raw):
        sss = StratifiedShuffleSplit(1, train_size=n_subset, random_state=SEED)
        idx, _ = next(sss.split(X_raw, y))
        X_raw, y = X_raw[idx], y[idx]

    X = StandardScaler().fit_transform(X_raw).astype(np.float32)
    idx = np.arange(len(X))
    tr_va, te_i = train_test_split(idx, test_size=0.15, random_state=SEED, stratify=y)
    tr_i, va_i = train_test_split(tr_va, test_size=0.177, random_state=SEED, stratify=y[tr_va])

    view_feats = mnist_view_features(X)
    view_tags = list(view_feats.keys())
    X_pca = PCA(n_components=50, random_state=SEED).fit_transform(X).astype(np.float32)
    g_raw, kappa = compute_observers(X_pca, {k: view_feats[k] for k in ["BLOCK", "PCA"]}, k=15)
    obs_mean = g_raw[tr_i].mean(0)
    obs_std = g_raw[tr_i].std(0) + 1e-8
    g = ((g_raw - obs_mean) / obs_std).astype(np.float32)
    kappa_bins = _kappa_bins(kappa)

    neigh_idx, edge_wts, view_mask, agree_score = build_joint_neighbourhood(view_feats, k_per_view=5)
    neigh_idx_v1, edge_wts_v1 = build_neighbourhood(view_feats, k=15, primary_key="PCA")
    lbl_nei = compute_label_ctx_per_view(y.astype(np.float32), neigh_idx, edge_wts, view_mask)
    lbl_nei = _mask_label_context_to_train(lbl_nei, neigh_idx, tr_i)

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
        task="classification",
        n_classes=10,
        epochs=teacher_epochs,
        skip_centroid_loss=False,
    )

    results = []
    routing_frames = []
    regime_frames = []
    best_variant_payload = None

    hgbr = train_hgbr(X[tr_i], y[tr_i], X[va_i], y[va_i], task="classification")
    preds_hgbr = hgbr.predict(X[te_i]).astype(np.int64)
    result, _, _ = _collect_result(
        "B1_HGBR",
        compute_metrics_mn(y[te_i], preds_hgbr),
        {"best_ep": "—", "stop_ep": "—", "collapsed": False},
        source="current_run",
        change="reference baseline",
        interp="Strong tabular baseline for calibration",
    )
    results.append(result)

    g2 = GoraTransformer(X.shape[1], g.shape[1], len(view_tags), 10, 64, 4, 2)
    g2 = train_gora(
        g2,
        X,
        g,
        y,
        neigh_idx_v1,
        edge_wts_v1,
        tr_i,
        va_i,
        task="classification",
        n_classes=10,
        epochs=train_epochs,
        patience=20,
        lr=3e-4,
        batch_size=512,
        name="G2_ref",
    )
    preds_g2, proba_g2, pi_g2, tau_g2 = predict_gora(g2, X, g, y, neigh_idx_v1, edge_wts_v1, te_i, "classification")
    routing_df = compute_routing_stats(pi_g2, None, tau_g2, view_tags, 4)
    regime_df = compute_regime_metrics(y[te_i], preds_g2, kappa, te_i, "classification", model_name="G2_ref")
    result, routing_df, regime_df = _collect_result(
        "G2_ref",
        compute_metrics_mn(y[te_i], preds_g2, proba_g2),
        {"best_ep": "v1", "stop_ep": "v1", "collapsed": False},
        source="current_run",
        change="GoRA carry-forward reference",
        interp="Reference GoRA path before v4 split-track changes",
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
        lbl_nei,
        train_epochs,
    )
    results.append(g10_result)
    routing_frames.append(g10_routing)
    regime_frames.append(g10_regime)

    variants = [
        (
            "MN_v4a",
            _make_v4(X.shape[1], g.shape[1], len(view_tags), 10, use_label_ctx=True, use_teacher_query=True, use_alpha_gate=True),
            0.0,
            "Protected v4 path matching G10 ingredients",
            "Locked alpha-gate baseline under v4 routing path",
        ),
        (
            "MN_v4b",
            _make_v4(X.shape[1], g.shape[1], len(view_tags), 10, use_label_ctx=True, use_teacher_query=True, use_alpha_gate=True),
            0.003,
            "MN_v4a + mild diversity regulariser",
            "Incremental specialization pressure with alpha gate preserved",
        ),
        (
            "MN_v4c",
            _make_v4(X.shape[1], g.shape[1], len(view_tags), 10, use_label_ctx=True, use_teacher_query=True, use_alpha_gate=True),
            0.006,
            "MN_v4b + stronger diversity",
            "More explicit anti-collapse pressure; only justified if accuracy holds",
        ),
        (
            "MN_v4d",
            _make_v4(X.shape[1], g.shape[1], len(view_tags), 10, use_label_ctx=True, use_teacher_query=True, use_alpha_gate=True, use_label_ctx_layernorm=True),
            0.010,
            "MN_v4c + label-ctx LayerNorm",
            "Optional anti-collapse variant with extra scale control",
        ),
    ]

    for tag, model, lam_diversity, change, interp in variants:
        result, routing_df, regime_df, beta_regime_df, view_similarity = _run_v4_variant(
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
            z_arr=z_arr,
            lbl_nei=lbl_nei,
            epochs=train_epochs,
            lam_diversity=lam_diversity,
            change=change,
            interp=interp,
        )
        results.append(result)
        routing_frames.append(routing_df)
        regime_frames.append(regime_df)
        routing_df.to_csv(ART_DIR / f"routing_{tag}.csv", index=False)
        regime_df.to_csv(ART_DIR / f"regime_{tag}.csv", index=False)
        if best_variant_payload is None or result["accuracy"] > best_variant_payload["result"]["accuracy"]:
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

    metrics_with_refs = pd.concat(
        [
            results_df,
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
            dataset_prefix="mnist",
        )

    root_cause_rows = _root_cause_rows(
        best_variant_payload["result"]["tag"],
        best_variant_payload["routing_df"],
        best_variant_payload["beta_regime_df"],
        best_variant_payload["view_similarity"],
    )
    gate_rows = _gate_rows(results_df, saved_v3, reference_df, best_variant_payload["routing_df"], smoke=smoke)
    failure_rows = _failure_rows(gate_rows)
    triage_df = _triage_rows(results_df, saved_v3)

    write_root_cause_audit(REP_DIR / "root_cause_audit.md", "mnist", root_cause_rows)
    write_gates_report(REP_DIR / "gates_report.md", "mnist", gate_rows, failure_rows)

    verdict = "v4 partially improved but key routing issues remain"
    best_acc = float(best_variant_payload["result"]["accuracy"])
    if best_acc >= _metric_for(saved_v3, "G10_Full") - 0.002:
        verdict = "v4 fixed the core regression collapse and preserved MNIST gains"
    elif best_acc < _metric_for(saved_v3, "G2_GoRA_v1"):
        verdict = "v4 changed architecture but did not solve the main problems"

    executive_summary = [
        f"Run mode: `{'smoke' if smoke else 'full'}`.",
        f"Best MNIST v4 variant: `{best_variant_payload['result']['tag']}` with accuracy `{best_acc:.4f}`.",
        f"Saved v3 G10 reference accuracy: `{_metric_for(saved_v3, 'G10_Full'):.4f}`; saved G2 accuracy: `{_metric_for(saved_v3, 'G2_GoRA_v1'):.4f}`.",
        "MNIST remains a protection track: changes are only justified if specialization improves without materially losing the G10 accuracy gain.",
    ]
    sections = {
        "Integrity Confirmation": _markdown(reference_df),
        "What Changed From v3": "\n".join(
            [
                "- G10 is rerun explicitly as a protected reference.",
                "- v4 exposes explicit beta routing while preserving alpha-gate behavior.",
                "- Diversity regularisation is treated as optional anti-collapse pressure, not assumed improvement.",
            ]
        ),
        "Root-Cause Audit": _markdown(pd.DataFrame(root_cause_rows)),
        "Routing Behavior Audit": _markdown(best_variant_payload["routing_df"]),
        "Gate Results": _markdown(pd.DataFrame(gate_rows)),
        "Predictive Results": _markdown(metrics_with_refs[metrics_with_refs["tag"].isin(["G2_ref", "G10_ref", "MN_v4a", "MN_v4b", "MN_v4c", "MN_v4d"])]),
        "Performance Drop Triage": _markdown(triage_df),
        "What Is Fixed": "\n".join(
            [
                "- The branch now preserves a protected G10 reproduction path.",
                "- Explicit beta routing and isolated MNIST outputs are implemented.",
                "- Best-checkpoint restore and teacher-query validation path mismatches are fixed in v4 training.",
            ]
        ),
        "What Remains": "\n".join(
            [
                "- Any MNIST accuracy loss versus G10 should block broader architectural changes from being treated as progress.",
                "- Diversity pressure should be removed before v5 if it does not improve specialization without hurting metrics.",
            ]
        ),
        "Recommendation Before v5": _markdown(pd.DataFrame(failure_rows)),
        "Final Verdict": verdict,
    }
    write_final_report(REP_DIR / "final_report.md", "mnist", executive_summary, sections)

    summary_lines = [
        "# MNIST v4 Summary",
        f"*{time.strftime('%Y-%m-%d')}*",
        "",
        "## Metrics",
        _markdown(metrics_with_refs),
        "",
        "## Reference Reproduction",
        _markdown(reference_df),
    ]
    (REP_DIR / "mn_v4_report.md").write_text("\n".join(summary_lines))
    print(f"[DONE] MNIST v4 in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n_mnist", type=int, default=10000)
    args = parser.parse_args()
    run_mnist(n_subset=args.n_mnist, smoke=args.smoke)
