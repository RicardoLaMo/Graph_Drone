"""
eval_v4.py — Evaluation, routing diagnostics, figures, and report helpers for MQ-GoRA v4.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def compute_metrics_ca(y_true, y_pred) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_metrics_mn(y_true, y_pred, proba=None) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, proba)) if proba is not None else float("nan"),
    }


def _normalized_entropy(pi_h: np.ndarray) -> float:
    n_views = pi_h.shape[-1]
    ent = -(pi_h * np.log(pi_h + 1e-8)).sum(axis=-1).mean()
    return float(ent / max(np.log(n_views), 1e-8))


def compute_routing_stats(
    pi_all: Optional[np.ndarray],
    beta_all: Optional[np.ndarray],
    tau_np: Optional[np.ndarray],
    view_tags: List[str],
    n_heads: int,
) -> pd.DataFrame:
    rows = []
    for head_idx in range(n_heads):
        row: Dict[str, object] = {"head_idx": head_idx}
        if pi_all is not None:
            pi_h = pi_all[:, head_idx, :]
            mean_pi = pi_h.mean(axis=0)
            top1 = pi_h.argmax(axis=-1)
            row["routing_entropy"] = _normalized_entropy(pi_h)
            row["dominant_view"] = view_tags[int(mean_pi.argmax())]
            for view_idx, view_name in enumerate(view_tags):
                row[f"mean_pi_{view_name}"] = float(mean_pi[view_idx])
                row[f"top1_freq_{view_name}"] = float((top1 == view_idx).mean())
        if beta_all is not None:
            beta_h = beta_all[:, head_idx]
            row["mean_beta"] = float(beta_h.mean())
            row["beta_std"] = float(beta_h.std())
        if tau_np is not None and head_idx < len(tau_np):
            row["tau"] = float(tau_np[head_idx])
        rows.append(row)
    return pd.DataFrame(rows)


def _named_regimes(values: np.ndarray, n_bins: int = 3):
    if n_bins != 3:
        return [f"bin_{i}" for i in range(n_bins)]
    return ["low", "medium", "high"]


def compute_regime_metrics(
    y_true,
    y_pred,
    kappa_all: np.ndarray,
    te_i: np.ndarray,
    task: str,
    model_name: str,
    n_bins: int = 3,
) -> pd.DataFrame:
    kappa_te = kappa_all[te_i]
    cuts = np.percentile(kappa_te, np.linspace(0, 100, n_bins + 1))
    cuts[0] -= 1e-6
    cuts[-1] += 1e-6
    bin_ids = np.digitize(kappa_te, cuts) - 1
    names = _named_regimes(kappa_te, n_bins=n_bins)
    rows = []
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if mask.sum() == 0:
            continue
        row = {
            "model": model_name,
            "regime": names[bin_idx],
            "kappa_lo": float(cuts[bin_idx]),
            "kappa_hi": float(cuts[bin_idx + 1]),
            "n": int(mask.sum()),
        }
        if task == "regression":
            row["rmse"] = float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
        else:
            row["accuracy"] = float(accuracy_score(y_true[mask], y_pred[mask]))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_beta_by_regime(
    beta_all: Optional[np.ndarray],
    kappa_all: np.ndarray,
    te_i: np.ndarray,
    n_bins: int = 3,
) -> pd.DataFrame:
    if beta_all is None:
        return pd.DataFrame(columns=["regime", "head_idx", "mean_beta"])
    kappa_te = kappa_all[te_i]
    cuts = np.percentile(kappa_te, np.linspace(0, 100, n_bins + 1))
    cuts[0] -= 1e-6
    cuts[-1] += 1e-6
    bin_ids = np.digitize(kappa_te, cuts) - 1
    names = _named_regimes(kappa_te, n_bins=n_bins)
    rows = []
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if mask.sum() == 0:
            continue
        for head_idx in range(beta_all.shape[1]):
            rows.append(
                {
                    "regime": names[bin_idx],
                    "head_idx": head_idx,
                    "mean_beta": float(beta_all[mask, head_idx].mean()),
                }
            )
    return pd.DataFrame(rows)


def compute_view_context_similarity(
    view_ctxs: Optional[np.ndarray],
    view_tags: List[str],
) -> pd.DataFrame:
    if view_ctxs is None:
        return pd.DataFrame(np.eye(len(view_tags)), index=view_tags, columns=view_tags)
    mean_ctx = view_ctxs.mean(axis=0)
    norms = np.linalg.norm(mean_ctx, axis=1, keepdims=True) + 1e-8
    normed = mean_ctx / norms
    sim = normed @ normed.T
    return pd.DataFrame(sim, index=view_tags, columns=view_tags)


def write_routing_figures(
    fig_dir: Path,
    routing_df: pd.DataFrame,
    beta_by_regime_df: pd.DataFrame,
    view_similarity: pd.DataFrame,
    dataset_prefix: str = "",
) -> None:
    del dataset_prefix
    fig_dir.mkdir(parents=True, exist_ok=True)

    mean_pi_cols = [c for c in routing_df.columns if c.startswith("mean_pi_")]
    if mean_pi_cols:
        matrix = routing_df[mean_pi_cols].to_numpy()
        xticks = [c.removeprefix("mean_pi_") for c in mean_pi_cols]
        fig, ax = plt.subplots(figsize=(max(4, len(mean_pi_cols) * 1.6), max(3, len(routing_df) * 0.8)))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            xticklabels=xticks,
            yticklabels=[f"Head {h}" for h in routing_df["head_idx"]],
            ax=ax,
        )
        ax.set_title("Mean pi by head")
        plt.tight_layout()
        plt.savefig(fig_dir / "pi_by_head.png", dpi=150)
        plt.close()

    top1_cols = [c for c in routing_df.columns if c.startswith("top1_freq_")]
    if top1_cols:
        plot_df = routing_df[["head_idx", *top1_cols]].melt(
            id_vars="head_idx",
            var_name="view",
            value_name="frequency",
        )
        plot_df["view"] = plot_df["view"].str.removeprefix("top1_freq_")
        fig, ax = plt.subplots(figsize=(max(5, len(top1_cols) * 1.6), 4))
        sns.barplot(plot_df, x="view", y="frequency", hue="head_idx", ax=ax)
        ax.set_title("Top-1 view frequency by head")
        ax.set_ylim(0.0, 1.0)
        plt.tight_layout()
        plt.savefig(fig_dir / "top1_view_freq.png", dpi=150)
        plt.close()

    if not beta_by_regime_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(beta_by_regime_df, x="regime", y="mean_beta", hue="head_idx", ax=ax)
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Mean beta by curvature regime")
        plt.tight_layout()
        plt.savefig(fig_dir / "beta_distribution.png", dpi=150)
        plt.close()

    fig, ax = plt.subplots(figsize=(max(4, view_similarity.shape[0] * 1.2), max(3, view_similarity.shape[0] * 0.9)))
    sns.heatmap(view_similarity, annot=True, fmt=".2f", cmap="crest", vmin=-1.0, vmax=1.0, ax=ax)
    ax.set_title("View-context similarity")
    plt.tight_layout()
    plt.savefig(fig_dir / "view_context_similarity.png", dpi=150)
    plt.close()

    if "tau" in routing_df.columns:
        fig, ax = plt.subplots(figsize=(max(4, len(routing_df) * 0.8), 3))
        ax.bar(routing_df["head_idx"], routing_df["tau"], color="#dd8452")
        ax.set_xlabel("Head")
        ax.set_ylabel("tau")
        ax.set_title("Per-head tau")
        plt.tight_layout()
        plt.savefig(fig_dir / "tau_distribution.png", dpi=150)
        plt.close()


def write_markdown_table(path: Path, title: str, rows: Iterable[Dict], intro: Optional[List[str]] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", f"*{date.today()}*", ""]
    if intro:
        lines.extend(intro)
        lines.append("")
    df = pd.DataFrame(list(rows))
    lines.append(df.to_markdown(index=False) if not df.empty else "_No rows_")
    path.write_text("\n".join(lines))


def write_v4_report(
    dataset: str,
    task: str,
    results: List[Dict],
    agree_score_mean: float,
    view_tags: List[str],
    report_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame([{**{"tag": r["tag"]}, **r["metrics"]} for r in results])
    lines = [
        f"# MQ-GoRA v4: {dataset.upper()}",
        f"*{date.today()}* | Branch: `feature/mq-gora-v4-split-track`",
        "",
        "## Metrics",
        metrics_df.to_markdown(index=False),
        "",
        f"Mean agree_score = {agree_score_mean:.3f}",
        "",
        f"Views: {', '.join(view_tags)}",
    ]
    report_path.write_text("\n".join(lines))


def write_root_cause_audit(report_path: Path, dataset: str, rows: List[Dict]) -> None:
    intro = [
        f"Dataset: `{dataset}`",
        "PASS / PARTIAL / FAIL reflects the v3 root-cause audit requirements.",
    ]
    write_markdown_table(report_path, "Root Cause Audit", rows, intro=intro)


def write_gates_report(report_path: Path, dataset: str, rows: List[Dict], failures: Optional[List[Dict]] = None) -> None:
    lines = [
        f"# Gates Report: {dataset}",
        f"*{date.today()}*",
        "",
        pd.DataFrame(rows).to_markdown(index=False) if rows else "_No gates_",
    ]
    if failures:
        lines += ["", "## Failure-to-Fix Mapping", "", pd.DataFrame(failures).to_markdown(index=False)]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))


def write_final_report(
    report_path: Path,
    dataset: str,
    executive_summary: List[str],
    sections: Dict[str, str],
) -> None:
    lines = [f"# Final Report: {dataset}", f"*{date.today()}*", ""]
    lines += ["## Executive Summary", *executive_summary, ""]
    for title, body in sections.items():
        lines += [f"## {title}", body, ""]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
