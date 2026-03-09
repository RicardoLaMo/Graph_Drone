"""
run_ca_v5.py — v5 California Housing experiment.

Ablation ladder (PRD 2026-03-08):
  A0  backbone_v5_ref    : HeadRoutingBackboneV5 + FlatRegressionHead (reproduces worktree behaviour)
  A1  v5_head_gated      : + HeadGatedRegressor (Gap 6) ← primary hypothesis test
  B0  v5_probe_a0        : freeze A0 backbone, HGBR on head_repr (baseline probe)
  B1  v5_probe           : freeze A1 backbone, fit HGBR on head_repr (diagnostic)
  C1  v5_per_view_obs    : + per-view quality prior q[B,V] into pi (Gap 1)
  C2  v5_jaccard_beta    : + Jaccard-anchored beta prior (Gap 2)
  C3  v5_sigma2          : + sigma2_v router input (Gap 13)
  D1  v5_adaptive_tau    : + row-adaptive geom_scale (Gap 3)
  D2  v5_pdiv_lb         : + L_pdiv + L_lb regularizers (Gap 4)
  E1  v5_orth            : A1 + L_orth=0.02 (attention head diversity, like HR_v2_diverse)
  E2  v5_orth_init       : E1 + head-specific view bias initialization (break symmetry)
  E3  v5_orth_pdiv       : E2 + L_pdiv=0.05 (prediction diversity on top of diversity)

Gate:  A1 RMSE < G2_ref=0.4546  ← primary blocker
Stretch: any C* or D* or E* beats HGBR=0.4433
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parents[1]
REPO_ROOT = EXP_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.head_routing_v5.california.src.data_v5 import build_california_dataset
from experiments.head_routing_v5.california.src.views_v5 import build_california_views, build_per_view_knn
from experiments.head_routing_v5.california.src.train_v5 import (
    TrainConfigV5,
    V5Model,
    get_device,
    maybe_truncate_splits,
    predict,
    standardise_observers,
    train,
)
from experiments.head_routing_v5.shared.src.backbone_v5 import HeadRoutingBackboneV5
from experiments.head_routing_v5.shared.src.observers_v5 import build_v5_observers
from experiments.head_routing_v5.shared.src.task_heads_v5 import (
    FlatRegressionHead,
    HeadGatedRegressor,
)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

N_HEADS = 4
HEAD_DIM = 16
ROUTER_HIDDEN = 64
HEAD_HIDDEN = 64   # GateNet hidden dim
DROPOUT = 0.1
KNN_K = 15


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output-root", type=Path, default=EXP_DIR)
    p.add_argument(
        "--experiments",
        nargs="+",
        default=["A0", "A1", "B0", "B1", "C1", "C2", "C3", "D1", "D2", "E1", "E2", "E3"],
        help="Which experiments to run"
    )
    return p.parse_args()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    resid = y_true - y_pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ---------------------------------------------------------------------------
# Build model for a given experiment variant
# ---------------------------------------------------------------------------

def build_model(
    view_input_dims: dict,
    obs_dim: int,
    n_views: int,
    # Backbone flags
    use_quality_prior: bool = False,
    use_jaccard_prior: bool = False,
    use_sigma2: bool = False,
    use_adaptive_tau: bool = False,
    use_view_embed: bool = False,
    # Head type
    use_head_gate: bool = False,
    # Initialization
    use_head_view_bias_init: bool = False,
) -> V5Model:
    backbone = HeadRoutingBackboneV5(
        view_input_dims=view_input_dims,
        obs_dim=obs_dim,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        router_hidden_dim=ROUTER_HIDDEN,
        dropout=DROPOUT,
        use_quality_prior=use_quality_prior,
        use_jaccard_prior=use_jaccard_prior,
        use_sigma2=use_sigma2,
        use_adaptive_tau=use_adaptive_tau,
        use_view_embed=use_view_embed,
    )
    n_view_pairs = n_views * (n_views - 1) // 2

    if use_head_gate:
        head = HeadGatedRegressor(
            n_heads=N_HEADS,
            head_dim=HEAD_DIM,
            obs_dim=obs_dim,
            n_views=n_views,
            hidden_dim=HEAD_HIDDEN,
            dropout=DROPOUT,
            use_quality_in_gate=use_quality_prior,
            use_global_residual=False,
            use_uncertainty=False,
        )
    else:
        head = FlatRegressionHead(
            n_heads=N_HEADS,
            head_dim=HEAD_DIM,
            hidden_dim=128,
            dropout=DROPOUT,
        )
    if use_head_view_bias_init:
        backbone.router.init_head_view_biases(strength=2.0)

    return V5Model(backbone, head)


# ---------------------------------------------------------------------------
# Routing analysis helpers
# ---------------------------------------------------------------------------

def analyze_routing(routing: dict, view_names: list, label: str) -> dict:
    pi = routing["pi"]    # [N, H, V]
    beta = routing["beta"]  # [N, H, 1]
    gate_h = routing.get("gate_h")  # [N, H] or None

    rows = {
        "model": label,
        "mean_beta": float(beta.mean()),
    }
    for v_idx, vn in enumerate(view_names):
        rows[f"mean_pi_{vn}"] = float(pi[:, :, v_idx].mean())
    if gate_h is not None:
        for h in range(gate_h.shape[1]):
            rows[f"mean_gate_h{h}"] = float(gate_h[:, h].mean())
        # Head prediction spread
        rows["gate_entropy"] = float(
            -(gate_h * np.log(gate_h + 1e-8)).sum(axis=-1).mean()
        )
    return rows


# ---------------------------------------------------------------------------
# Backbone probe (B1): freeze backbone, HGBR on head_repr
# ---------------------------------------------------------------------------

def run_probe(
    model: V5Model,
    view_feats: dict,
    per_view_knn: dict,
    g_scaled: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    target_stats: dict,
    quality_score, quality_norm_arr, J_flat, mean_J, sigma2_v,
    batch_size: int = 512,
) -> dict:
    """Extract head_repr features from the frozen backbone and fit HGBR."""
    device = get_device()
    model = model.to(device).eval()
    dummy_y = np.zeros(next(iter(view_feats.values())).shape[0], dtype=np.float32)

    def get_head_repr(indices):
        from experiments.head_routing_v5.california.src.train_v5 import fetch_batch
        reprs = []
        for start in range(0, len(indices), batch_size):
            bi = indices[start: start + batch_size]
            batch = fetch_batch(
                bi, view_feats, per_view_knn, g_scaled, dummy_y, device,
                quality_score=quality_score,
                quality_norm_arr=quality_norm_arr,
                J_flat=J_flat,
                mean_J=mean_J,
                sigma2_v=sigma2_v,
            )
            with torch.no_grad():
                head_repr, _ = model.backbone(
                    x_anchor_by_view=batch["x_anchor_by_view"],
                    x_nei_by_view=batch["x_nei_by_view"],
                    ew_by_view=batch["ew_by_view"],
                    g=batch["g"],
                    quality_score=batch.get("quality_score"),
                    quality_norm=batch.get("quality_norm"),
                    J_flat=batch.get("J_flat"),
                    mean_J=batch.get("mean_J"),
                    sigma2_v=batch.get("sigma2_v"),
                )
            reprs.append(head_repr.cpu().numpy().reshape(len(bi), -1))
        return np.concatenate(reprs, axis=0)

    X_train_feat = get_head_repr(train_idx)
    X_test_feat = get_head_repr(test_idx)

    probe = HistGradientBoostingRegressor(
        max_iter=300, max_depth=6, learning_rate=0.05,
        random_state=42, n_iter_no_change=15,
    )
    probe.fit(X_train_feat, y[train_idx])
    probe_preds = probe.predict(X_test_feat).astype(np.float32)
    return compute_metrics(y[test_idx], probe_preds)


# ---------------------------------------------------------------------------
# HGBR baseline
# ---------------------------------------------------------------------------

def run_hgbr(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
    model = HistGradientBoostingRegressor(
        max_iter=300, max_depth=6, learning_rate=0.05,
        random_state=42, n_iter_no_change=15,
    )
    model.fit(X[train_idx], y[train_idx])
    preds = model.predict(X[test_idx]).astype(np.float32)
    return compute_metrics(y[test_idx], preds)


# ---------------------------------------------------------------------------
# Write results
# ---------------------------------------------------------------------------

def write_results(
    metrics_rows: list[dict],
    routing_rows: list[dict],
    output_root: Path,
):
    art_dir = output_root / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(art_dir / "metrics_v5.csv", index=False)

    if routing_rows:
        routing_df = pd.DataFrame(routing_rows)
        routing_df.to_csv(art_dir / "routing_v5.csv", index=False)

    # Print summary
    print("\n" + "=" * 72)
    print("CA v5 RESULTS")
    print("=" * 72)
    cols = ["model", "rmse", "mae", "r2", "best_epoch", "stop_epoch"]
    available = [c for c in cols if c in metrics_df.columns]
    print(metrics_df[available].to_string(index=False))
    print("=" * 72)

    # Gate checks
    g2_ref = 0.4546
    hgbr_ref = 0.4433
    a1_row = metrics_df[metrics_df["model"] == "A1_head_gated"]
    if len(a1_row):
        a1_rmse = float(a1_row["rmse"].values[0])
        gate = "✅ PASS" if a1_rmse < g2_ref else "❌ FAIL"
        print(f"\nGate C1 (A1 RMSE < G2={g2_ref}): A1={a1_rmse:.4f}  {gate}")
        if a1_rmse < hgbr_ref:
            print(f"🎯 STRETCH GOAL: A1 beats HGBR ({a1_rmse:.4f} < {hgbr_ref})")

    for key in ["C1", "C2", "C3", "D1", "D2"]:
        row = metrics_df[metrics_df["model"].str.startswith(key)]
        if len(row):
            rmse = float(row["rmse"].values[0])
            if rmse < hgbr_ref:
                print(f"🎯 STRETCH GOAL: {key} beats HGBR ({rmse:.4f} < {hgbr_ref})")

    return metrics_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(42)
    run_exps = set(args.experiments)

    output_root = args.output_root
    smoke = args.smoke
    max_epochs = 5 if smoke else 150
    patience = 3 if smoke else 30

    # ---- Data ----
    print("Loading California Housing dataset...")
    dataset = build_california_dataset(seed=42)
    X, y = dataset["X"], dataset["y"]
    train_idx, val_idx, test_idx = maybe_truncate_splits(
        dataset["train_idx"], dataset["val_idx"], dataset["test_idx"],
        smoke=smoke,
    )
    target_stats = dataset["target_stats"]
    print(f"  Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # ---- Views & kNN ----
    print("Building views and per-view kNN...")
    view_feats = build_california_views(X, train_idx=train_idx)
    per_view_knn = build_per_view_knn(view_feats, k=KNN_K, train_idx=train_idx)
    view_names = list(view_feats.keys())
    view_input_dims = {name: arr.shape[1] for name, arr in view_feats.items()}
    n_views = len(view_names)

    # ---- Global observers (g) ----
    print("Computing v5 observers (this may take a few minutes)...")
    obs_v5 = build_v5_observers(
        view_feats=view_feats,
        per_view_knn=per_view_knn,
        y=y,        # needed for sigma2_v
        train_idx=train_idx,
        k=KNN_K,
    )
    g_global = obs_v5["g_global"]
    g_scaled, _ = standardise_observers(g_global, train_idx)
    obs_dim = g_scaled.shape[1]

    quality_score = obs_v5["quality_score"]     # [N, V]
    quality_norm_arr = obs_v5["quality_norm"]   # [N, V, 3]
    J_flat = obs_v5["J_flat"]                   # [N, n_pairs]
    mean_J = obs_v5["mean_J"]                   # [N]
    sigma2_v = obs_v5["sigma2_v"]               # [N, V] or None

    print(f"  obs_dim={obs_dim}, quality_score range: [{quality_score.min():.3f}, {quality_score.max():.3f}]")
    print(f"  J_flat range: [{J_flat.min():.3f}, {J_flat.max():.3f}], mean={J_flat.mean():.3f}")
    if sigma2_v is not None:
        print(f"  sigma2_v range: [{sigma2_v.min():.3f}, {sigma2_v.max():.3f}]")

    config_base = TrainConfigV5(
        max_epochs=max_epochs,
        patience=patience,
        batch_size=512,
        lr=3e-4,
        weight_decay=1e-4,
        loss_type="huber",
    )

    metrics_rows = []
    routing_rows = []

    # ---- HGBR baseline (always) ----
    print("\n[Baseline] HGBR...")
    hgbr_metrics = run_hgbr(X, y, train_idx, test_idx)
    metrics_rows.append({"model": "B1_HGBR", **hgbr_metrics, "best_epoch": "—", "stop_epoch": "—"})
    print(f"  HGBR RMSE={hgbr_metrics['rmse']:.4f}")

    def _run(label, backbone_flags, use_head_gate, config_overrides=None,
             use_head_view_bias_init=False, extras=None):
        """Run one experiment variant."""
        set_seed(42)
        cfg = config_base
        if config_overrides:
            cfg = TrainConfigV5(**{**cfg.__dict__, **config_overrides})

        print(f"\n[{label}] backbone_flags={backbone_flags}, head_gate={use_head_gate}"
              f"{', head_init=True' if use_head_view_bias_init else ''}")
        model = build_model(
            view_input_dims=view_input_dims,
            obs_dim=obs_dim,
            n_views=n_views,
            use_head_gate=use_head_gate,
            use_head_view_bias_init=use_head_view_bias_init,
            **backbone_flags,
        )

        # Which v5 arrays to pass
        qs = quality_score if backbone_flags.get("use_quality_prior") or use_head_gate else None
        qn = quality_norm_arr if backbone_flags.get("use_quality_prior") else None
        jf = J_flat if backbone_flags.get("use_jaccard_prior") else None
        mj = mean_J if backbone_flags.get("use_jaccard_prior") else None
        sv = sigma2_v if backbone_flags.get("use_sigma2") else None

        model = train(
            model, view_feats, per_view_knn, g_scaled, y,
            train_idx, val_idx, target_stats, cfg,
            quality_score=qs,
            quality_norm_arr=qn,
            J_flat=jf,
            mean_J=mj,
            sigma2_v=sv,
        )
        preds, routing = predict(
            model, test_idx, view_feats, per_view_knn, g_scaled, target_stats,
            quality_score=qs,
            quality_norm_arr=qn,
            J_flat=jf,
            mean_J=mj,
            sigma2_v=sv,
        )
        metrics = compute_metrics(y[test_idx], preds)
        meta = getattr(model, "training_metadata", {})
        row = {
            "model": label,
            **metrics,
            "best_epoch": meta.get("best_epoch", "—"),
            "stop_epoch": meta.get("stop_epoch", "—"),
            "collapsed": meta.get("collapsed", False),
        }
        metrics_rows.append(row)
        routing_rows.append(analyze_routing(routing, view_names, label))
        print(f"  RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  R²={metrics['r2']:.4f}"
              f"  best_epoch={meta.get('best_epoch','—')}")
        return model, routing, qs, qn, jf, mj, sv

    # ---- A0: reference (flat head, no quality prior) ----
    a0_model = a1_model = None
    a0_qs = a0_qn = a0_jf = a0_mj = a0_sv = None
    a1_qs = a1_qn = a1_jf = a1_mj = a1_sv = None

    if "A0" in run_exps:
        a0_model, _, a0_qs, a0_qn, a0_jf, a0_mj, a0_sv = _run(
            "A0_ref", {}, use_head_gate=False
        )

    # ---- A1: head-gated (Gap 6) ----
    if "A1" in run_exps:
        a1_model, _, a1_qs, a1_qn, a1_jf, a1_mj, a1_sv = _run(
            "A1_head_gated", {}, use_head_gate=True
        )

    # ---- B0: backbone probe on A0 (baseline — how good is A0's backbone?) ----
    if "B0" in run_exps and a0_model is not None:
        print("\n[B0] Backbone probe on A0 backbone (baseline: what is A0 head_repr worth?)...")
        probe_metrics = run_probe(
            a0_model, view_feats, per_view_knn, g_scaled, y,
            train_idx, test_idx, target_stats,
            a0_qs, a0_qn, a0_jf, a0_mj, a0_sv,
        )
        metrics_rows.append({"model": "B0_probe_a0", **probe_metrics, "best_epoch": "—", "stop_epoch": "—"})
        print(f"  Probe RMSE={probe_metrics['rmse']:.4f}")
        a0_rmse = next((r["rmse"] for r in metrics_rows if r["model"] == "A0_ref"), None)
        if a0_rmse:
            gap = probe_metrics["rmse"] - a0_rmse
            print(f"  Gap (probe - A0_e2e) = {gap:.4f}  {'→ A0 flat head is efficient' if gap < 0.01 else '→ HGBR probe beats A0 flat head'}")

    # ---- B1: backbone probe on A1 ----
    if "B1" in run_exps and a1_model is not None:
        print("\n[B1] Backbone probe (frozen A1 backbone + HGBR on head_repr)...")
        probe_metrics = run_probe(
            a1_model, view_feats, per_view_knn, g_scaled, y,
            train_idx, test_idx, target_stats,
            a1_qs, a1_qn, a1_jf, a1_mj, a1_sv,
        )
        metrics_rows.append({"model": "B1_probe", **probe_metrics, "best_epoch": "—", "stop_epoch": "—"})
        print(f"  Probe RMSE={probe_metrics['rmse']:.4f}")
        a1_rmse = next((r["rmse"] for r in metrics_rows if r["model"] == "A1_head_gated"), None)
        if a1_rmse:
            gap = a1_rmse - probe_metrics["rmse"]
            print(f"  Gap (A1_e2e - probe) = {gap:.4f}  {'→ decoder was bottleneck' if gap > 0.01 else '→ backbone is the bottleneck'}")

    # ---- C1: per-view quality prior ----
    if "C1" in run_exps:
        _run("C1_per_view_obs",
             {"use_quality_prior": True},
             use_head_gate=True)

    # ---- C2: + Jaccard-anchored beta prior ----
    if "C2" in run_exps:
        _run("C2_jaccard_beta",
             {"use_quality_prior": True, "use_jaccard_prior": True},
             use_head_gate=True)

    # ---- C3: + sigma2_v ----
    if "C3" in run_exps:
        if sigma2_v is not None:
            _run("C3_sigma2",
                 {"use_quality_prior": True, "use_jaccard_prior": True, "use_sigma2": True},
                 use_head_gate=True)
        else:
            print("[C3] sigma2_v not available (y was not passed) — skipping")

    # ---- D1: adaptive tau (Gap 3) ----
    if "D1" in run_exps:
        _run("D1_adaptive_tau",
             {"use_quality_prior": True, "use_jaccard_prior": True, "use_adaptive_tau": True},
             use_head_gate=True)

    # ---- D2: + L_pdiv + L_lb ----
    if "D2" in run_exps:
        _run("D2_pdiv_lb",
             {"use_quality_prior": True, "use_jaccard_prior": True, "use_adaptive_tau": True},
             use_head_gate=True,
             config_overrides={"lambda_pdiv": 0.05, "lambda_lb": 0.01})

    # ---- E1: A1 + L_orth (attention diversity, like HR_v2_diverse) ----
    # Hypothesis: diversity loss forces head_repr differentiation → gate can work
    if "E1" in run_exps:
        _run("E1_orth",
             {},
             use_head_gate=True,
             config_overrides={"lambda_orth": 0.02})

    # ---- E2: E1 + head-specific view bias init (break symmetry at t=0) ----
    if "E2" in run_exps:
        _run("E2_orth_init",
             {},
             use_head_gate=True,
             use_head_view_bias_init=True,
             config_overrides={"lambda_orth": 0.02})

    # ---- E3: E2 + L_pdiv=0.05 (prediction diversity on top of representation diversity) ----
    if "E3" in run_exps:
        _run("E3_orth_pdiv",
             {},
             use_head_gate=True,
             use_head_view_bias_init=True,
             config_overrides={"lambda_orth": 0.02, "lambda_pdiv": 0.05})

    # ---- Write results ----
    write_results(metrics_rows, routing_rows, output_root)
    print(f"\nArtifacts saved to: {output_root / 'artifacts'}")


if __name__ == "__main__":
    main()
