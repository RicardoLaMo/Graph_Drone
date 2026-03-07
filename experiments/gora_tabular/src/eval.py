"""
eval.py — Evaluation + head specialisation analysis for GoRA-Tabular.

Required outputs:
  - metrics.csv: per-model predictive metrics
  - routing_stats.csv: head-view affinity matrix, entropy, tau
  - figures: pi_heatmap.png, head_view_affinity.png, per_bin_metric.png, tau_distribution.png
"""
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error,
                              mean_absolute_error, r2_score, log_loss)
from scipy.stats import entropy as scipy_entropy
from typing import Dict, List
import datetime


def score(name, y_true, y_pred, task, y_proba=None):
    if task == "regression":
        rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        print(f"  [{name}] RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
        return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}
    else:
        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        ll = float(log_loss(y_true, y_proba)) if y_proba is not None else float("nan")
        print(f"  [{name}] Acc={acc:.4f} F1={f1:.4f} LogLoss={ll:.4f}")
        return {"model": name, "accuracy": acc, "macro_f1": f1, "log_loss": ll}


def head_specialisation(pi_all: np.ndarray, view_tags: List[str]) -> pd.DataFrame:
    """
    pi_all: [N, H, M]
    Returns head-view affinity matrix [H, M] + routing entropy per head.
    Analogous to GoRA Table 5.
    """
    H, M = pi_all.shape[1], pi_all.shape[2]
    rows = []
    for h in range(H):
        pi_h = pi_all[:, h, :]          # [N, M]
        mean_pi = pi_h.mean(0)          # [M]
        top1 = pi_h.argmax(-1)
        top1_freq = np.bincount(top1, minlength=M) / len(top1)
        ent = scipy_entropy(mean_pi + 1e-9)
        dominant_view = view_tags[mean_pi.argmax()]
        row = {"head_idx": h, "entropy": float(ent), "dominant_view": dominant_view}
        for vi, vname in enumerate(view_tags):
            row[f"mean_pi_{vname}"] = float(mean_pi[vi])
            row[f"top1_freq_{vname}"] = float(top1_freq[vi])
        rows.append(row)
    return pd.DataFrame(rows)


def bin_metric(preds, y_true, kappa_bins, task, model_name):
    rows = []
    for b in ["low", "medium", "high"]:
        mask = kappa_bins == b
        if mask.sum() == 0: continue
        if task == "regression":
            val = float(mean_squared_error(y_true[mask], preds[mask]) ** 0.5)
            rows.append({"model": model_name, "bin": b, "rmse": val, "n": int(mask.sum())})
        else:
            val = float(accuracy_score(y_true[mask], preds[mask]))
            rows.append({"model": model_name, "bin": b, "accuracy": val, "n": int(mask.sum())})
    return rows


# ─── Figures ──────────────────────────────────────────────────────────────────

def fig_head_affinity(spec_df: pd.DataFrame, view_tags: List[str], fig_dir: Path, name: str):
    """Head-view affinity heatmap (GoRA Table 5 equivalent)."""
    cols = [f"mean_pi_{v}" for v in view_tags]
    matrix = spec_df[cols].values    # [H, M]
    fig, ax = plt.subplots(figsize=(max(4, len(view_tags) * 1.5), max(3, len(spec_df) * 0.8)))
    sns.heatmap(matrix, ax=ax, xticklabels=view_tags,
                yticklabels=[f"Head {h}" for h in spec_df.head_idx.values],
                annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"Head-View Affinity (mean π_{{i,h,m}}) — {name}")
    ax.set_xlabel("View"); ax.set_ylabel("Head")
    plt.tight_layout(); plt.savefig(fig_dir / f"head_view_affinity_{name}.png", dpi=150); plt.close()


def fig_pi_spread(pi_all: np.ndarray, view_tags: List[str], fig_dir: Path, name: str):
    """Distribution of routing weights per view across all rows, head 0."""
    pi_h0 = pi_all[:, 0, :]    # [N, M]
    fig, axes = plt.subplots(1, len(view_tags),
                              figsize=(3.5 * len(view_tags), 3.5), sharey=True)
    if len(view_tags) == 1: axes = [axes]
    for ax, vi, vn in zip(axes, range(len(view_tags)), view_tags):
        ax.hist(pi_h0[:, vi], bins=40, color="#4C72B0", alpha=0.8, edgecolor="none")
        ax.set_title(f"π[head0, {vn}]"); ax.set_xlabel("weight"); ax.set_xlim(0, 1)
    plt.suptitle(f"Routing weight distributions — {name}"); plt.tight_layout()
    plt.savefig(fig_dir / f"pi_distribution_{name}.png", dpi=150); plt.close()


def fig_tau(tau: np.ndarray, fig_dir: Path, name: str):
    """Per-head temperature τ_h."""
    fig, ax = plt.subplots(figsize=(max(4, len(tau) * 0.8), 3))
    ax.bar(range(len(tau)), tau, color="#DD8452")
    ax.set_xlabel("Head"); ax.set_ylabel("τ (temperature)"); ax.set_title(f"Per-head τ — {name}")
    plt.tight_layout(); plt.savefig(fig_dir / f"tau_{name}.png", dpi=150); plt.close()


def fig_per_bin(bin_rows: list, task: str, fig_dir: Path, name: str):
    df = pd.DataFrame(bin_rows)
    if df.empty: return
    metric = "rmse" if task == "regression" else "accuracy"
    models = df.model.unique()
    bins = ["low", "medium", "high"]
    x = np.arange(3); w = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    palette = sns.color_palette("muted", len(models))
    for idx, m in enumerate(models):
        sub = df[df.model == m]
        vals = [sub[sub.bin == b][metric].values[0] if (sub.bin == b).any() else 0 for b in bins]
        ax.bar(x + idx * w - 0.4 + w / 2, vals, w, label=m, color=palette[idx])
    ax.set_xticks(x); ax.set_xticklabels(["Low κ", "Med κ", "High κ"])
    ax.set_ylabel(metric); ax.set_title(f"Per-bin metric — {name}")
    ax.legend(fontsize=7); plt.tight_layout()
    plt.savefig(fig_dir / f"per_bin_{name}.png", dpi=150); plt.close()


# ─── Report ───────────────────────────────────────────────────────────────────

def write_report(name, task, metrics, spec_df, tau, view_tags, n_heads, rep_path):
    ts = datetime.datetime.now().strftime("%Y-%m-%d")
    m_df = pd.DataFrame(metrics)

    g2 = m_df[m_df.model == "G2_GoRA"]
    g3 = m_df[m_df.model == "G3_Uniform"]
    b1 = m_df[m_df.model == "B1_HGBR"]

    key = "rmse" if task == "regression" else "accuracy"
    better = lambda a, b: (a < b) if task == "regression" else (a > b)

    g2v = g2[key].values[0] if len(g2) else None
    g3v = g3[key].values[0] if len(g3) else None
    b1v = b1[key].values[0] if len(b1) else None

    routing_helps = g2v and g3v and better(g2v, g3v)
    gora_beats_baseline = g2v and b1v and better(g2v, b1v)

    g2s = f"{g2v:.4f}" if g2v is not None else "N/A"
    g3s = f"{g3v:.4f}" if g3v is not None else "N/A"
    b1s = f"{b1v:.4f}" if b1v is not None else "N/A"

    lines = [
        f"# GoRA-Tabular: {name}",
        f"*{ts}* | Branch: `feature/gora-tabular-routing`\n",
        "## Architecture",
        "- **Routing semantics:** π_{i,h,m} from MoERouter(g_i) shapes attention logits:",
        "  `logit_{ij}^h = <q^h,k^h>/√d + log(τ_h · Ã_{ij}^{i,h} + ε)`",
        "- **Routing is inside the softmax**, not a downstream combiner.",
        "- **Isolation vs interaction** is structural: a peaked π confines the softmax to",
        "  one view's neighbourhood; flat π spans all views.",
        f"- **Routing scope (this pass):** per-row, per-head (H={n_heads} heads).\n",
        "## Models",
        "| ID | Description |",
        "|----|-------------|",
        "| B0 | MLP |",
        "| B1 | HGBR |",
        "| G0 | Standard Transformer (no graph) |",
        f"| G1 | Single-view Transformer (fixed best view) |",
        "| **G2** | **GoRA-Tabular (full routing)** |",
        "| G3 | Uniform-π ablation (no geometry) |",
        "| G4 | Shuffled-g ablation (geometry destroyed) |\n",
        "## Metrics\n",
        m_df.to_markdown(index=False),
        "\n## Head-View Affinity (GoRA Table 5 equivalent)\n",
        spec_df.to_markdown(index=False),
        f"\n## Per-head temperature τ (learned): {[f'{t:.3f}' for t in tau]}\n",
        "## Audit Conclusion",
        f"- G2 routing beats G3 uniform? **{'YES' if routing_helps else 'NO'}** "
        f"(G2={g2s}, G3={g3s} {key})",
        f"- G2 beats strong tabular baseline (B1)? **{'YES' if gora_beats_baseline else 'NO'}** "
        f"(G2={g2s}, B1={b1s})",
        "",
        "## What makes this different from prior experiments",
        "- Prior: frozen reps → ObserverRouter → reweighting (post-hoc, Model A pattern)",
        "- This: g_i → π_{i,h,m} → logit bias inside softmax → representation formation",
        "- The graph neighbourhood each head sees is determined BEFORE embedding is complete.",
    ]
    rep_path.write_text("\n".join(lines))
    print(f"[report] Saved: {rep_path}")
