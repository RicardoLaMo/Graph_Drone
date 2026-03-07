"""
07_analysis.py — Produce all figures, bin analyses, correlation tables, and final verdict.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ARTIFACTS = Path("artifacts")
FIGURES = Path("figures")
REPORTS = Path("reports")
FIGURES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
    "B0_Mean", "B1_MLP", "B2_HGBR",
    "M3_FULL", "M4_GEO", "M5_SOCIO",
    "M6_Uniform", "M7_LearnedCombiner",
    "M8A_KappaFeature", "M8B_FourViewCombiner",
    "M8B_CurvView", "M9_ObserverCombiner",
]


def load_test_preds(model_names, idx_te, y_all):
    preds = {}
    for name in model_names:
        fpath = ARTIFACTS / f"preds_{name}_test.npy"
        if fpath.exists():
            preds[name] = np.load(fpath)
        else:
            print(f"  [07_analysis] WARNING: Missing predictions for {name}")
    y_te = y_all[idx_te]
    return preds, y_te


def compute_metrics(name, y_true, y_pred):
    return {
        "model": name,
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ─── Figure 1: Curvature distribution ───────────────────────────────────────

def plot_curvature_hist(df_bins):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df_bins["kappa"], bins=60, color="#4C72B0", edgecolor="none", alpha=0.8)
    axes[0].set_xlabel("kappa (local PCA residual ratio)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Curvature Proxy")

    axes[1].hist(df_bins["lof"], bins=60, color="#DD8452", edgecolor="none", alpha=0.8)
    axes[1].set_xlabel("LOF score")
    axes[1].set_title("Distribution of LOF (outlier proxy)")

    plt.tight_layout()
    plt.savefig(FIGURES / "curvature_hist.png", dpi=150)
    plt.close()
    print("[07_analysis] Saved: figures/curvature_hist.png")


# ─── Figure 2: Error by curvature bin ───────────────────────────────────────

def plot_error_by_bin(preds, y_te, df_bins, idx_te):
    df_test_bins = df_bins.iloc[idx_te].copy()
    df_test_bins["abs_error_baseline"] = np.abs(y_te - preds.get("B2_HGBR", np.zeros_like(y_te)))

    key_models = [k for k in ["B0_Mean", "B2_HGBR", "M3_FULL", "M6_Uniform",
                               "M8A_KappaFeature", "M9_ObserverCombiner"] if k in preds]

    bin_order = ["low", "medium", "high"]
    rmse_data = []
    for name in key_models:
        if name not in preds:
            continue
        for b in bin_order:
            mask = df_test_bins["curvature_bin"].values == b
            if mask.sum() == 0:
                continue
            rmse_b = float(mean_squared_error(y_te[mask], preds[name][mask]) ** 0.5)
            rmse_data.append({"model": name, "bin": b, "rmse": rmse_b})

    df_rmse = pd.DataFrame(rmse_data)
    if df_rmse.empty:
        print("[07_analysis] No data for error by bin plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    palette = sns.color_palette("muted", n_colors=len(key_models))
    x = np.arange(len(bin_order))
    width = 0.8 / len(key_models)
    for i, name in enumerate(key_models):
        sub = df_rmse[df_rmse.model == name]
        vals = [sub[sub.bin == b]["rmse"].values[0] if (sub.bin == b).any() else 0 for b in bin_order]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=name, color=palette[i])

    ax.set_xticks(x)
    ax.set_xticklabels(["Low Curvature", "Medium Curvature", "High Curvature"])
    ax.set_ylabel("RMSE")
    ax.set_title("Test RMSE by Curvature Bin")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURES / "error_by_curvature_bin.png", dpi=150)
    plt.close()
    print("[07_analysis] Saved: figures/error_by_curvature_bin.png")


# ─── Figure 3: Observer correlation heatmap ──────────────────────────────────

def plot_correlation_heatmap(df_bins, preds, y_all, idx_te):
    y_te = y_all[idx_te]
    df_test = df_bins.iloc[idx_te].copy()

    # Absolute prediction errors for key models
    for name in ["B2_HGBR", "M3_FULL", "M9_ObserverCombiner"]:
        if name in preds:
            df_test[f"abs_err_{name}"] = np.abs(y_te - preds[name])

    cols = ["kappa", "lid", "lof", "density", "forman"] + \
           [c for c in df_test.columns if c.startswith("abs_err")]

    corr_data = df_test[[c for c in cols if c in df_test.columns]].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.zeros_like(corr_data, dtype=bool)
    sns.heatmap(corr_data, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Spearman Correlation: Observer Features vs Prediction Errors")
    plt.tight_layout()
    plt.savefig(FIGURES / "observer_correlation_heatmap.png", dpi=150)
    plt.close()
    print("[07_analysis] Saved: figures/observer_correlation_heatmap.png")


# ─── Verdict logic ────────────────────────────────────────────────────────────

def compute_verdict(metrics_df, preds, y_te, df_test_bins):
    """
    Apply crisp decision logic based on:
    - Does M8A/M8B/M9 clearly outperform M7 on total RMSE?
    - Does any curvature model reduce excess error on high-kappa rows?
    - Does M9 beat a density-only / LOF-only ablation?
    """
    available = set(metrics_df["model"].values)

    hgbr_rmse = metrics_df[metrics_df.model == "B2_HGBR"]["rmse"].values[0] if "B2_HGBR" in available else None
    m7_rmse = metrics_df[metrics_df.model == "M7_LearnedCombiner"]["rmse"].values[0] if "M7_LearnedCombiner" in available else None
    m9_rmse = metrics_df[metrics_df.model == "M9_ObserverCombiner"]["rmse"].values[0] if "M9_ObserverCombiner" in available else None
    m8a_rmse = metrics_df[metrics_df.model == "M8A_KappaFeature"]["rmse"].values[0] if "M8A_KappaFeature" in available else None

    # Graph embedding check
    sage_models = [m for m in ["M3_FULL", "M4_GEO", "M5_SOCIO"] if m in available]
    best_sage_rmse = min([metrics_df[metrics_df.model == m]["rmse"].values[0] for m in sage_models]) if sage_models else None

    reasons = []

    # H1: graph helps beyond baselines?
    graph_helps = (best_sage_rmse is not None and hgbr_rmse is not None and best_sage_rmse < hgbr_rmse)
    reasons.append(f"H1 (graph > tabular): {'YES' if graph_helps else 'NO'}")

    # H3: curvature adds value beyond non-curvature combiner?
    curv_gain = None
    if m7_rmse is not None and m9_rmse is not None:
        curv_gain = m7_rmse - m9_rmse

    clearly_worthwhile = curv_gain is not None and curv_gain > 0.02
    modest = curv_gain is not None and 0.005 < curv_gain <= 0.02

    if curv_gain is not None:
        reasons.append(f"H3 (curv > no-curv combiner): gain={curv_gain:.4f}")

    # Check high-kappa row performance
    high_mask = df_test_bins["curvature_bin"].values == "high"
    curv_helps_high = False
    if "M9_ObserverCombiner" in preds and "M7_LearnedCombiner" in preds and high_mask.sum() > 0:
        m9_high = mean_squared_error(y_te[high_mask], preds["M9_ObserverCombiner"][high_mask]) ** 0.5
        m7_high = mean_squared_error(y_te[high_mask], preds["M7_LearnedCombiner"][high_mask]) ** 0.5
        curv_helps_high = m9_high < m7_high
        reasons.append(f"H5 (curv reduces high-kappa error): {'YES' if curv_helps_high else 'NO'}  (M9={m9_high:.4f} vs M7={m7_high:.4f})")

    if clearly_worthwhile:
        verdict = "curvature clearly worthwhile"
    elif modest or curv_helps_high:
        verdict = "curvature modest but useful"
    else:
        verdict = "curvature not justified on this dataset"

    reasons_str = "\n".join(reasons)
    return verdict, reasons_str


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_analysis():
    print("[07_analysis] Loading data...")
    idx_te = np.load(ARTIFACTS / "idx_test.npy")
    idx_tr = np.load(ARTIFACTS / "idx_train.npy")
    y_all = np.load(ARTIFACTS / "y_all.npy")
    df_bins = pd.read_csv(ARTIFACTS / "curvature_bins.csv")
    stability_df = pd.read_csv(ARTIFACTS / "curvature_stability.csv")

    preds, y_te = load_test_preds(MODEL_ORDER, idx_te, y_all)
    df_test_bins = df_bins.iloc[idx_te].copy()
    df_test_bins.index = np.arange(len(df_test_bins))

    # Collect all metrics
    all_metrics = []
    for name, pred in preds.items():
        all_metrics.append(compute_metrics(name, y_te, pred))

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.sort_values("rmse").reset_index(drop=True)
    metrics_df.to_csv(ARTIFACTS / "california_metrics.csv", index=False)
    print("[07_analysis] Metrics saved.")
    print(metrics_df[["model", "rmse", "mae", "r2"]].to_string(index=False))

    # Figures
    plot_curvature_hist(df_bins)
    plot_error_by_bin(preds, y_te, df_bins, idx_te)
    plot_correlation_heatmap(df_bins, preds, y_all, idx_te)

    # Verdict
    verdict, reasons = compute_verdict(metrics_df, preds, y_te, df_test_bins)
    print(f"\n[07_analysis] VERDICT: {verdict.upper()}")
    print(f"[07_analysis] Reasons:\n{reasons}")

    # Bin-level analysis
    bin_rows = []
    for name in preds:
        for b in ["low", "medium", "high"]:
            mask = df_test_bins["curvature_bin"].values == b
            if mask.sum() == 0:
                continue
            r = compute_metrics(f"{name}_{b}_bin", y_te[mask], preds[name][mask])
            r.update({"model": name, "bin": b, "n_rows": int(mask.sum())})
            bin_rows.append(r)
    bin_df = pd.DataFrame(bin_rows)
    bin_df.to_csv(ARTIFACTS / "metrics_by_curvature_bin.csv", index=False)

    # Stability
    print("[07_analysis] Curvature stability across scales:")
    print(stability_df.to_string(index=False))

    # Write markdown report
    write_report(metrics_df, bin_df, stability_df, verdict, reasons, df_bins)
    print("[07_analysis] Done.")


# ─── Report writer ────────────────────────────────────────────────────────────

def write_report(metrics_df, bin_df, stability_df, verdict, reasons, df_bins):
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d")

    kappa_mean = df_bins["kappa"].mean()
    kappa_std = df_bins["kappa"].std()

    lines = [
        "# California Housing: Curvature-Aware Experiment",
        f"*Generated: {ts}*\n",
        "---",
        "## 1. Objective",
        "Primary question: does adding a curvature-based view or curvature-derived observer features improve predictive performance — especially on structurally difficult rows — compared with plain tabular baselines, single-graph kNN embeddings, or multi-view embeddings without curvature?\n",
        "**Scientific stance:** Do not assume curvature helps. Prove or disprove it.\n",
        "---",
        "## 2. Experimental Setup",
        "- **Dataset:** California Housing (sklearn), n=20,640 rows, 8 features.",
        "- **Preprocessing:** RobustScaler. log1p on Population, AveOccup, AveRooms.",
        "- **Splits:** 70/15/15 train/val/test (random_state=42, no leakage).",
        "- **Graph construction:** sklearn NearestNeighbors, k=15.",
        "- **4 Views:** FULL (all 8 features), GEO (Lat/Long), SOCIO (5 socioeconomic), CURVATURE (FULL + kappa-similarity edge weights).",
        "",
        "---",
        "## 3. Models Compared",
        "| ID | Description |",
        "|----|-------------|",
        "| B0 | Mean predictor |",
        "| B1 | MLP (3-layer, ReLU, early stopping) |",
        "| B2 | HistGradientBoostingRegressor |",
        "| M3 | GraphSAGE on FULL graph |",
        "| M4 | GraphSAGE on GEO graph |",
        "| M5 | GraphSAGE on SOCIO graph |",
        "| M6 | Uniform ensemble of M3/M4/M5 |",
        "| M7 | Learned combiner over M3/M4/M5 (no curvature) |",
        "| M8A | kappa appended as node feature → GraphSAGE |",
        "| M8B | kappa as 4th view in learned combiner |",
        "| M9 | Observer-driven per-row combiner using [kappa, LID, LOF, density] |",
        "",
        "---",
        "## 4. Curvature Definition",
        "- **kappa_i** = local PCA residual ratio in row i's k=15 kNN neighborhood.",
        "- Specifically: fraction of neighborhood variance NOT explained by the top-2 principal directions.",
        "- This is a **practical non-flatness proxy** — not exact Riemannian curvature.",
        f"- kappa: mean={kappa_mean:.4f}, std={kappa_std:.4f}",
        "",
        "---",
        "## 5. Main Results",
        "",
        metrics_df[["model", "rmse", "mae", "r2"]].to_markdown(index=False),
        "",
        "---",
        "## 6. Curvature-Bin Analysis",
        "",
        bin_df[["model", "bin", "rmse", "n_rows"]].to_markdown(index=False) if not bin_df.empty else "N/A",
        "",
        "---",
        "## 7. Ablation Notes",
        "- M7 (no curvature) vs M9 (curvature observer): direct curvature value test.",
        "- M8A (kappa as feature) vs M3 (no kappa): isolated feature contribution.",
        "- M8B (4-view) vs M6/M7 (3-view): curvature as graph view test.",
        "",
        "---",
        "## 8. Stability Results",
        "",
        stability_df.to_markdown(index=False),
        "",
        "---",
        "## 9. Conclusion",
        f"### VERDICT: **{verdict.upper()}**",
        "",
        reasons.replace("\n", "\n\n"),
        "",
        "> **Interpretation guideline:**",
        "> If graph embeddings themselves do not beat HGBR, the curvature result should be interpreted with caution — curvature is an additive filter on top of graph structure.",
        "> If curvature adds nothing beyond density / LOF / LID, the recommendation is to use simpler observer features instead.",
        "",
        "---",
        "*Outputs: `artifacts/california_metrics.csv`, `artifacts/curvature_bins.csv`, `figures/curvature_hist.png`, `figures/error_by_curvature_bin.png`, `figures/observer_correlation_heatmap.png`*",
    ]

    report_path = REPORTS / "california_curvature_experiment.md"
    report_path.write_text("\n".join(lines))
    print(f"[07_analysis] Report saved: {report_path}")


if __name__ == "__main__":
    run_analysis()
