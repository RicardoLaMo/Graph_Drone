"""
s07_analysis.py — Figures, bin analysis, hypothesis evaluation, and report.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix

EXP_ROOT = Path(__file__).parent.parent
ARTIFACTS = EXP_ROOT / "artifacts"
FIGURES = EXP_ROOT / "figures"
REPORTS = EXP_ROOT / "reports"
FIGURES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
    "M0_Majority", "M1_MLP", "M2_HGBR", "M3_XGBoost",
    "M4_TabPFN_subset",
    "M5_FULL", "M6_BLOCK", "M7_PCA",
    "M8_Uniform", "M9_Combiner",
    "M10_KappaFeature", "M11_ObsCombiner",
]

PRED_MAP = {
    "M0_Majority": "preds_M0_Majority_test.npy",
    "M1_MLP": "preds_M1_MLP_test.npy",
    "M2_HGBR": "preds_M2_HGBR_test.npy",
    "M3_XGBoost": "preds_M3_XGB_test.npy",
    "M4_TabPFN_subset": "preds_M4_TabPFN_test_subset.npy",
    "M5_FULL": "preds_M5_FULL_test.npy",
    "M6_BLOCK": "preds_M6_BLOCK_test.npy",
    "M7_PCA": "preds_M7_PCA_test.npy",
    "M8_Uniform": "preds_M8_Uniform_test.npy",
    "M9_Combiner": "preds_M9_Combiner_test.npy",
    "M10_KappaFeature": "preds_M10_KappaFeature_test.npy",
    "M11_ObsCombiner": "preds_M11_ObsCombiner_test.npy",
}

PROBA_MAP = {
    "M1_MLP": "proba_M1_MLP_test.npy",
    "M2_HGBR": "proba_M2_HGBR_test.npy",
    "M3_XGBoost": "proba_M3_XGB_test.npy",
}


def load_preds(idx_te, y_all):
    y_te = y_all[idx_te]
    preds, probas = {}, {}
    for name, fname in PRED_MAP.items():
        p = ARTIFACTS / fname
        if p.exists():
            arr = np.load(p)
            # TabPFN is on a subset — align to its own test slice
            if name == "M4_TabPFN_subset":
                preds[name] = arr  # kept as-is, labelled separately
            else:
                preds[name] = arr
    for name, fname in PROBA_MAP.items():
        p = ARTIFACTS / fname
        if p.exists():
            probas[name] = np.load(p)
    return preds, probas, y_te


def compute_metrics(name, y_true, y_pred, y_proba=None):
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    ll = float(log_loss(y_true, y_proba)) if y_proba is not None else float("nan")
    return {"model": name, "accuracy": acc, "macro_f1": f1, "log_loss": ll}


# ── Figure 1: Curvature distribution ─────────────────────────────────────────

def plot_curvature_hist(df_bins):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df_bins["kappa"], bins=60, color="#4C72B0", edgecolor="none", alpha=0.8)
    axes[0].set_xlabel("kappa (local PCA residual)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Curvature Proxy Distribution (MNIST rows)")
    axes[1].hist(df_bins["lof"], bins=60, color="#DD8452", edgecolor="none", alpha=0.8)
    axes[1].set_xlabel("LOF score")
    axes[1].set_title("LOF Distribution")
    plt.tight_layout()
    plt.savefig(FIGURES / "curvature_hist.png", dpi=150)
    plt.close()
    print("[s07] curvature_hist.png saved")


# ── Figure 2: Accuracy by curvature bin ──────────────────────────────────────

def plot_error_by_bin(preds, y_te, df_test_bins):
    key = [m for m in ["M1_MLP", "M3_XGBoost", "M5_FULL", "M9_Combiner",
                        "M11_ObsCombiner"] if m in preds]
    bins = ["low", "medium", "high"]
    rows = []
    for name in key:
        for b in bins:
            mask = df_test_bins["curvature_bin"].values == b
            if mask.sum() == 0: continue
            rows.append({"model": name, "bin": b,
                         "accuracy": float(accuracy_score(y_te[mask], preds[name][mask]))})
    df = pd.DataFrame(rows)
    if df.empty: return
    fig, ax = plt.subplots(figsize=(12, 5))
    pal = sns.color_palette("muted", n_colors=len(key))
    x = np.arange(3); w = 0.8 / len(key)
    for i, name in enumerate(key):
        sub = df[df.model == name]
        vals = [sub[sub.bin == b]["accuracy"].values[0] if (sub.bin == b).any() else 0 for b in bins]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=name, color=pal[i])
    ax.set_xticks(x); ax.set_xticklabels(["Low κ", "Medium κ", "High κ"])
    ax.set_ylabel("Accuracy"); ax.set_title("Accuracy by Curvature Bin")
    ax.legend(fontsize=7); plt.tight_layout()
    plt.savefig(FIGURES / "error_by_curvature_bin.png", dpi=150)
    plt.close()
    print("[s07] error_by_curvature_bin.png saved")


# ── Figure 3: View comparison bar chart ──────────────────────────────────────

def plot_view_comparison(metrics_df):
    sub = metrics_df[metrics_df.model.isin(["M1_MLP","M3_XGBoost","M5_FULL","M6_BLOCK",
                                             "M7_PCA","M8_Uniform","M9_Combiner",
                                             "M10_KappaFeature","M11_ObsCombiner"])].copy()
    sub = sub.sort_values("accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4C72B0" if "M5" in r or "M6" in r or "M7" in r or "M8" in r or "M9" in r
              else "#DD8452" if "M10" in r or "M11" in r
              else "#55A868" for r in sub["model"]]
    ax.barh(sub["model"], sub["accuracy"], color=colors)
    ax.set_xlabel("Test Accuracy"); ax.set_title("Model Comparison — MNIST Geometry")
    ax.axvline(sub[sub.model=="M1_MLP"]["accuracy"].values[0] if "M1_MLP" in sub.model.values else 0,
               color="red", ls="--", lw=1, label="MLP baseline")
    ax.legend(fontsize=8); plt.tight_layout()
    plt.savefig(FIGURES / "view_comparison.png", dpi=150)
    plt.close()
    print("[s07] view_comparison.png saved")


# ── Figure 4: Stability heatmap ───────────────────────────────────────────────

def plot_stability_heatmap(stab_df):
    keys = ["spearman_k10_k20", "spearman_k10_k30", "spearman_k20_k30",
            "top20pct_overlap_k10_k20", "top20pct_overlap_k10_k30", "top20pct_overlap_k20_k30"]
    vals = [stab_df[k].values[0] for k in keys if k in stab_df.columns]
    labels = ["ρ k10-k20", "ρ k10-k30", "ρ k20-k30", "top20% k10-k20", "top20% k10-k30", "top20% k20-k30"]
    fig, ax = plt.subplots(figsize=(8, 3))
    cmap = plt.cm.RdYlGn
    for i, (v, lbl) in enumerate(zip(vals, labels)):
        c = cmap(v)
        ax.barh(lbl, v, color=c)
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.1); ax.set_title("Curvature Stability Across k-Scales"); plt.tight_layout()
    plt.savefig(FIGURES / "stability_heatmap.png", dpi=150)
    plt.close()
    print("[s07] stability_heatmap.png saved")


# ── Hypothesis evaluation ─────────────────────────────────────────────────────

def evaluate_hypotheses(metrics_df, preds, y_te, df_test_bins):
    avail = set(metrics_df["model"].values)

    def get_acc(m): return metrics_df[metrics_df.model == m]["accuracy"].values[0] if m in avail else None

    mlp = get_acc("M1_MLP"); xgb = get_acc("M3_XGBoost")
    best_sage = max([get_acc(m) for m in ["M5_FULL","M6_BLOCK","M7_PCA"] if get_acc(m) is not None], default=None)
    m9_acc = get_acc("M9_Combiner"); m11_acc = get_acc("M11_ObsCombiner")
    m8_acc = get_acc("M8_Uniform")

    H = {}
    H["H1"] = best_sage is not None and mlp is not None and best_sage > mlp
    H["H2"] = m8_acc is not None and best_sage is not None and m8_acc > best_sage * 0.99
    H["H3"] = m11_acc is not None and m9_acc is not None and (m11_acc - m9_acc) > 0.002

    high_mask = df_test_bins["curvature_bin"].values == "high"
    if "M11_ObsCombiner" in preds and "M9_Combiner" in preds and high_mask.sum() > 0:
        m11_high = accuracy_score(y_te[high_mask], preds["M11_ObsCombiner"][high_mask])
        m9_high = accuracy_score(y_te[high_mask], preds["M9_Combiner"][high_mask])
        H["H5"] = m11_high > m9_high
    else:
        H["H5"] = None

    H["H7"] = best_sage is not None and xgb is not None and best_sage > xgb

    reasons = [f"H1 (graph > MLP): {'YES' if H['H1'] else 'NO'} — best_sage={best_sage:.4f}, mlp={mlp:.4f}" if mlp and best_sage else "H1: N/A",
               f"H2 (multi-view > single): {'YES' if H['H2'] else 'NO'}",
               f"H3 (curv adds value): {'YES' if H['H3'] else 'NO'} — M11={m11_acc:.4f} M9={m9_acc:.4f}" if m11_acc and m9_acc else "H3: N/A",
               f"H5 (curv helps high-kappa): {'YES' if H.get('H5') else 'NO/N/A'}",
               f"H7 (best graph > XGBoost): {'YES' if H['H7'] else 'NO'}" if best_sage and xgb else "H7: N/A"]

    all_yes = all(v for v in [H.get("H1"), H.get("H3"), H.get("H5")] if v is not None)
    h1_yes = H.get("H1", False)
    h3_yes = H.get("H3", False)

    if all_yes:
        verdict = "Mechanism signal is present"
    elif h1_yes and not h3_yes:
        verdict = "Graph signal present, curvature signal weak"
    elif h1_yes and h3_yes:
        verdict = "Mechanism signal is present"
    elif not h1_yes:
        verdict = "Warning sign: hidden geometry was not meaningfully exploited"
    else:
        verdict = "Curvature not justified on MNIST under current implementation"

    return verdict, "\n".join(reasons), H


# ── Report writer ─────────────────────────────────────────────────────────────

def write_report(metrics_df, bin_df, stab_df, verdict, reasons, kappa_mean, kappa_std):
    import datetime; ts = datetime.datetime.now().strftime("%Y-%m-%d")
    lines = [
        "# MNIST-784: Geometry Sanity Check Report",
        f"*Generated: {ts}*  |  Branch: `feature/mnist-geometry-sanity-check`\n",
        "> ⚠️ This report is isolated from the California Housing experiment. No prior files modified.\n",
        "## 1. Mission",
        "Test whether kNN + graph embedding + curvature-view pipeline detects and exploits hidden pixel geometry in flattened MNIST-784.\n",
        "## 2. Main Results\n",
        metrics_df[["model","accuracy","macro_f1","log_loss"]].to_markdown(index=False),
        "\n## 3. Curvature Statistics",
        f"- kappa mean={kappa_mean:.4f}, std={kappa_std:.4f}",
        f"- {'kappa has useful variance' if kappa_std > 0.01 else 'WARNING: kappa near-zero variance'}\n",
        "## 4. Multi-Scale Stability\n",
        stab_df.to_markdown(index=False) if not stab_df.empty else "N/A",
        "\n## 5. Per-Bin Accuracy\n",
        bin_df[["model","bin","accuracy","n_rows"]].to_markdown(index=False) if not bin_df.empty else "N/A",
        "\n## 6. Hypothesis Results",
        f"```\n{reasons}\n```\n",
        "## 7. XGBoost & TabPFN Comparison",
        "- TabPFN was run on a ≤1024-sample subset (documented public limit). Result not directly comparable to full-dataset models.",
        "- XGBoost was evaluated on the full train/test split.\n",
        "## 8. Conclusion",
        f"### VERDICT: **{verdict.upper()}**",
        "",
        "> If graph embeddings do not beat MLP on MNIST, that is a strong warning sign—",
        "> MNIST has well-known hidden geometry that a proper graph pipeline should exploit.\n",
        "*Outputs: `artifacts/metrics.csv`, `figures/curvature_hist.png`, `figures/view_comparison.png`*",
    ]
    path = REPORTS / "mnist_geometry_sanity_report.md"
    path.write_text("\n".join(lines))
    print(f"[s07] Report saved: {path}")


def run_analysis():
    print("[s07] Loading data...")
    idx_te = np.load(ARTIFACTS / "idx_test.npy")
    idx_tr = np.load(ARTIFACTS / "idx_train.npy")
    y_all = np.load(ARTIFACTS / "y_all.npy")
    df_bins = pd.read_csv(ARTIFACTS / "curvature_bins.csv")
    stab_df = pd.read_csv(ARTIFACTS / "curvature_stability.csv")
    kappa = np.load(ARTIFACTS / "kappa.npy")

    preds, probas, y_te = load_preds(idx_te, y_all)
    df_test_bins = df_bins.iloc[idx_te].copy().reset_index(drop=True)

    # Metrics
    all_metrics = []
    for name, pred in preds.items():
        if name == "M4_TabPFN_subset":
            # evaluated on its own subset
            y_sub = y_all[idx_te[:len(pred)]]
            all_metrics.append(compute_metrics(name, y_sub, pred))
        else:
            pr = probas.get(name)
            all_metrics.append(compute_metrics(name, y_te, pred, pr))

    metrics_df = pd.DataFrame(all_metrics).sort_values("accuracy", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(ARTIFACTS / "metrics.csv", index=False)
    print("[s07] Metrics:"); print(metrics_df[["model","accuracy","macro_f1"]].to_string(index=False))

    # Figures
    plot_curvature_hist(df_bins)
    plot_error_by_bin(preds, y_te, df_test_bins)
    plot_view_comparison(metrics_df)
    plot_stability_heatmap(stab_df)

    # Hypotheses + verdict
    verdict, reasons, H = evaluate_hypotheses(metrics_df, preds, y_te, df_test_bins)
    print(f"\n[s07] VERDICT: {verdict.upper()}")
    print(f"[s07] Reasons:\n{reasons}")

    # Bin accuracy
    bin_rows = []
    for name in [m for m in MODEL_ORDER if m in preds and m != "M4_TabPFN_subset"]:
        for b in ["low","medium","high"]:
            mask = df_test_bins["curvature_bin"].values == b
            if mask.sum() == 0: continue
            r = compute_metrics(name, y_te[mask], preds[name][mask])
            r.update({"model": name, "bin": b, "n_rows": int(mask.sum())})
            bin_rows.append(r)
    bin_df = pd.DataFrame(bin_rows)
    bin_df.to_csv(ARTIFACTS / "metrics_by_bin.csv", index=False)

    write_report(metrics_df, bin_df, stab_df, verdict, reasons, float(kappa.mean()), float(kappa.std()))
    print("[s07] Done.")


if __name__ == "__main__":
    run_analysis()
