"""
eval_v4.py — Evaluation, routing diagnostics, and report generation for v4.

Functions:
  compute_metrics_ca     — RMSE, MAE, R2
  compute_metrics_mn     — accuracy, macro-F1, log-loss
  compute_routing_stats  — pi entropy, dominant view, tau, alpha stats
  write_v4_report        — unified markdown report writer
"""
import sys, os

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3_SRC = os.path.normpath(os.path.join(_HERE, '..', '..', '..', 'gora_tabular', 'src'))
if _V3_SRC not in sys.path:
    sys.path.insert(0, _V3_SRC)

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, log_loss,
)
from typing import Dict, List, Optional


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics_ca(y_true, y_pred) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def compute_metrics_mn(y_true, y_pred, proba=None) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    ll  = float(log_loss(y_true, proba)) if proba is not None else float("nan")
    return {"accuracy": acc, "macro_f1": f1, "log_loss": ll}


# ─── Routing diagnostics ─────────────────────────────────────────────────────

def compute_routing_stats(
    pi_all: Optional[np.ndarray],      # [N, H, M]
    tau_np: Optional[np.ndarray],      # [H]
    view_tags: List[str],
    n_heads: int,
) -> pd.DataFrame:
    """Return a DataFrame with per-head routing statistics."""
    rows = []
    for h in range(n_heads):
        row: Dict = {"head_idx": h}
        if pi_all is not None:
            pi_h = pi_all[:, h, :]                # [N, M]
            entropy = float(-(pi_h * np.log(pi_h + 1e-8)).sum(-1).mean())
            dom_idx = int(pi_h.mean(0).argmax())
            row["entropy"] = entropy
            row["dominant_view"] = view_tags[dom_idx]
            for vi, vt in enumerate(view_tags):
                row[f"mean_pi_{vt}"] = float(pi_h[:, vi].mean())
                row[f"top1_freq_{vt}"] = float((pi_h.argmax(-1) == vi).mean())
        if tau_np is not None:
            row["tau"] = float(tau_np[h]) if h < len(tau_np) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def compute_regime_metrics(
    y_true, y_pred, kappa_all: np.ndarray, te_i: np.ndarray,
    task: str, n_bins: int = 4,
) -> pd.DataFrame:
    """Per-curvature-bin performance."""
    kappa_te = kappa_all[te_i]
    bins = np.percentile(kappa_te, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-6; bins[-1] += 1e-6
    bin_ids = np.digitize(kappa_te, bins) - 1
    rows = []
    for b in range(n_bins):
        mask = (bin_ids == b)
        if mask.sum() < 5:
            continue
        row = {"kappa_bin": b, "n": int(mask.sum()),
               "kappa_lo": float(bins[b]), "kappa_hi": float(bins[b + 1])}
        if task == "regression":
            row["rmse"] = float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
        else:
            row["accuracy"] = float(accuracy_score(y_true[mask], y_pred[mask]))
        rows.append(row)
    return pd.DataFrame(rows)


# ─── Report writers ───────────────────────────────────────────────────────────

def write_v4_report(
    dataset: str,           # "california" | "mnist"
    task: str,
    results: List[Dict],    # list of {"tag": str, "metrics": dict, "best_ep": int, "stop_ep": int, "routing": DataFrame}
    agree_score_mean: float,
    view_tags: List[str],
    report_path: Path,
) -> None:
    """Write the unified v4 markdown report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ds_upper = dataset.upper()
    metric_key = "rmse" if task == "regression" else "accuracy"
    lower_is_better = (task == "regression")

    lines = [
        f"# MQ-GoRA v4 (Split-Track): {ds_upper}",
        f"*{date.today()}* | Branch: `feature/mq-gora-v4-split-track`",
        "",
        "## Architecture: MQGoraTransformerV4",
        "Split-track design — CA uses regression-safe routing; MNIST uses full v3+.",
        "",
        "## Metrics",
        "",
    ]

    if task == "regression":
        header = "| model | rmse | mae | r2 | best_ep | stop_ep | collapsed |"
        sep    = "|:------|-----:|----:|---:|--------:|--------:|----------:|"
        lines += [header, sep]
        for r in results:
            m = r["metrics"]
            collapsed = "Y" if r.get("collapsed", False) else "N"
            lines.append(
                f"| {r['tag']} | {m.get('rmse', float('nan')):.4f} |"
                f" {m.get('mae', float('nan')):.4f} |"
                f" {m.get('r2', float('nan')):.4f} |"
                f" {r.get('best_ep', '?')} | {r.get('stop_ep', '?')} | {collapsed} |"
            )
    else:
        header = "| model | accuracy | macro_f1 | log_loss | best_ep | stop_ep |"
        sep    = "|:------|--------:|---------:|---------:|--------:|--------:|"
        lines += [header, sep]
        for r in results:
            m = r["metrics"]
            lines.append(
                f"| {r['tag']} | {m.get('accuracy', float('nan')):.4f} |"
                f" {m.get('macro_f1', float('nan')):.4f} |"
                f" {m.get('log_loss', float('nan')):.4f} |"
                f" {r.get('best_ep', '?')} | {r.get('stop_ep', '?')} |"
            )

    lines += ["", "## View agreement score", f"Mean agree_score = {agree_score_mean:.3f}", ""]

    # California ablation table
    if task == "regression":
        lines += [
            "## CA v4 Ablation Table",
            "",
            "| Gate | Change | RMSE | best_ep | stop_ep | Collapsed | Interpretation |",
            "|------|--------|-----:|--------:|--------:|----------:|----------------|",
        ]
        ref_rmse = None
        for r in results:
            tag = r['tag']
            rmse = r['metrics'].get('rmse', float('nan'))
            be = r.get('best_ep', '?')
            se = r.get('stop_ep', '?')
            col = "Y" if r.get('collapsed', False) else "N"
            interp = r.get('interp', '—')
            change = r.get('change', '—')
            lines.append(f"| {tag} | {change} | {rmse:.4f} | {be} | {se} | {col} | {interp} |")

    # Head routing stats per model
    lines += ["", "## Head-View Affinity by model", ""]
    for r in results:
        df = r.get("routing")
        if df is None or df.empty:
            continue
        lines.append(f"### {r['tag']}")
        lines.append(df.to_markdown(index=False))
        lines.append("")

    report_path.write_text("\n".join(lines))
    print(f"[report v4] Saved: {report_path}")
