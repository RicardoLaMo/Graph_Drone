"""
TabArena Curated Classification Benchmark
==========================================
Compares GraphDrone-HyperSetRouter (fixed) vs GraphDrone-Baseline vs TabPFN
on classification datasets drawn from the TabArena curated suite.

Metrics: F1-macro, AUC-ROC (OvR macro), PR-AUC (macro)
Ref: https://github.com/TabArena/tabarena_dataset_curation
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# TabArena curated classification datasets (OpenML task IDs)
# These are the standard binary + multiclass classification datasets in the
# TabArena benchmark suite.
# --------------------------------------------------------------------------- #
TABARENA_CLASSIFICATION_TASKS = {
    # Binary
    "credit-g":         31,
    "adult":            7592,
    "blood-transfusion": 10101,
    "bank-marketing":   14965,
    "PhishingWebsites": 9952,
    "kc1":              3917,
    "churn":            3573,
    "spam":             44,
    "cylinder-bands":   14954,
    "diabetes":         37,
    # Multiclass
    "wine":             40691,
    "glass":            41,
    "vehicle":          54,
    "segment":          40984,
    "mnist-small":      3573,   # replaced below if load fails
    "mfeat-factors":    12,
    "letter":           6,
    "pendigits":        32,
    "optdigits":        28,
    "satimage":         182,
}

# Limit to a manageable subset for wall-clock budget
# Note: letter (task 6, 26 classes) excluded — exceeds TabPFN's 10-class limit
# wine 40691 → use task 40975 (OpenML wine quality); segment → task 146212
DEFAULT_TASKS = [
    ("credit-g",          31),
    ("adult",             7592),
    ("blood-transfusion", 10101),
    ("bank-marketing",    14965),
    ("PhishingWebsites",  9952),
    ("diabetes",          37),
    ("vehicle",           54),
    ("segment",           146212),
    ("mfeat-factors",     12),
    ("pendigits",         32),
    ("optdigits",         28),
]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray, n_classes: int) -> dict:
    from sklearn.metrics import (
        f1_score, roc_auc_score, average_precision_score
    )
    from sklearn.preprocessing import label_binarize

    metrics = {}

    # F1-macro
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # AUC-ROC (OvR macro)
    try:
        if n_classes == 2:
            pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            metrics["auc_roc"] = roc_auc_score(y_true, pos)
        else:
            metrics["auc_roc"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
    except Exception:
        metrics["auc_roc"] = float("nan")

    # PR-AUC (macro over classes)
    try:
        if n_classes == 2:
            pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            metrics["pr_auc"] = average_precision_score(y_true, pos)
        else:
            y_bin = label_binarize(y_true, classes=sorted(np.unique(y_true)))
            pr_aucs = []
            for c in range(y_bin.shape[1]):
                try:
                    pr_aucs.append(average_precision_score(y_bin[:, c], y_proba[:, c]))
                except Exception:
                    pass
            metrics["pr_auc"] = float(np.mean(pr_aucs)) if pr_aucs else float("nan")
    except Exception:
        metrics["pr_auc"] = float("nan")

    return metrics


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_openml_task(task_id: int, max_samples: int = 5000):
    import openml
    task = openml.tasks.get_task(task_id, download_data=True)
    X, y = task.get_X_and_y()
    X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

    # Encode categoricals
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = pd.factorize(X[col])[0].astype(np.float32)
    X = X.fillna(X.median(numeric_only=True)).astype(np.float32)
    X = X.values

    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    n_classes = len(np.unique(y))

    # Subsample for speed (stratified)
    if len(X) > max_samples:
        from sklearn.model_selection import train_test_split
        # Use train_test_split as a stratified subsampler
        _, X, _, y = train_test_split(
            X, y, test_size=max_samples, random_state=42, stratify=y
        )

    return X, y, n_classes


# --------------------------------------------------------------------------- #
# Model runners
# --------------------------------------------------------------------------- #
def run_graphdrone_hyper(X_tr, y_tr, X_te, y_te, n_classes):
    """GraphDrone with HyperSetRouter (TaskToken wired) and Multi-Expert Portfolio."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from graphdrone_fit.model import GraphDrone
    from graphdrone_fit.config import GraphDroneConfig, SetRouterConfig
    from graphdrone_fit.expert_factory import ExpertBuildSpec, IdentitySelectorAdapter, PcaProjectionAdapter
    from graphdrone_fit.view_descriptor import ViewDescriptor

    n_feats = X_tr.shape[1]
    full_idx = tuple(range(n_feats))
    
    # Diversity Portfolio: FULL + PCA(50%) + PCA(25%)
    params = {"n_estimators": 8, "device": "cuda", "n_classes": n_classes}
    pca1_dim = max(1, n_feats // 2)
    pca2_dim = max(1, n_feats // 4)

    specs = (
        ExpertBuildSpec(
            descriptor=ViewDescriptor(expert_id="FULL", family="FULL", view_name="Full", is_anchor=True, input_dim=n_feats, input_indices=full_idx),
            model_kind="foundation_classifier", input_adapter=IdentitySelectorAdapter(indices=full_idx), model_params=params
        ),
        ExpertBuildSpec(
            descriptor=ViewDescriptor(expert_id="PCA_MID", family="structural_subspace", view_name=f"PCA_{pca1_dim}", input_dim=pca1_dim, input_indices=full_idx, projection_kind="structural_subspace"),
            model_kind="foundation_classifier", input_adapter=PcaProjectionAdapter(n_components=pca1_dim), model_params=params
        ),
        ExpertBuildSpec(
            descriptor=ViewDescriptor(expert_id="PCA_LOW", family="structural_subspace", view_name=f"PCA_{pca2_dim}", input_dim=pca2_dim, input_indices=full_idx, projection_kind="structural_subspace"),
            model_kind="foundation_classifier", input_adapter=PcaProjectionAdapter(n_components=pca2_dim), model_params=params
        )
    )

    cfg = GraphDroneConfig(
        router=SetRouterConfig(kind="hyper_set_router"),
        problem_type="classification",
        n_classes=n_classes,
    )
    gd = GraphDrone(cfg)
    gd.fit(X_tr, y_tr, expert_specs=specs)

    result = gd.predict(X_te, return_diagnostics=True)
    y_proba = result.predictions
    if y_proba.ndim == 1:
        y_proba = np.stack([1 - y_proba, y_proba], axis=1)
    y_pred = np.argmax(y_proba, axis=1)

    return y_pred, y_proba, result.diagnostics


def run_graphdrone_baseline(X_tr, y_tr, X_te, y_te, n_classes):
    """GraphDrone with BootstrapFullRouter and Multi-Expert Portfolio (routing disabled)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from graphdrone_fit.model import GraphDrone
    from graphdrone_fit.config import GraphDroneConfig, SetRouterConfig
    from graphdrone_fit.expert_factory import ExpertBuildSpec, IdentitySelectorAdapter, PcaProjectionAdapter
    from graphdrone_fit.view_descriptor import ViewDescriptor

    n_feats = X_tr.shape[1]
    full_idx = tuple(range(n_feats))
    
    # Diversity Portfolio: FULL + PCA(50%) + PCA(25%)
    params = {"n_estimators": 8, "device": "cuda", "n_classes": n_classes}
    pca1_dim = max(1, n_feats // 2)
    pca2_dim = max(1, n_feats // 4)

    specs = (
        ExpertBuildSpec(
            descriptor=ViewDescriptor(expert_id="FULL", family="FULL", view_name="Full", is_anchor=True, input_dim=n_feats, input_indices=full_idx),
            model_kind="foundation_classifier", input_adapter=IdentitySelectorAdapter(indices=full_idx), model_params=params
        ),
        ExpertBuildSpec(
            descriptor=ViewDescriptor(expert_id="PCA_MID", family="structural_subspace", view_name=f"PCA_{pca1_dim}", input_dim=pca1_dim, input_indices=full_idx, projection_kind="structural_subspace"),
            model_kind="foundation_classifier", input_adapter=PcaProjectionAdapter(n_components=pca1_dim), model_params=params
        ),
        ExpertBuildSpec(
            descriptor=ViewDescriptor(expert_id="PCA_LOW", family="structural_subspace", view_name=f"PCA_{pca2_dim}", input_dim=pca2_dim, input_indices=full_idx, projection_kind="structural_subspace"),
            model_kind="foundation_classifier", input_adapter=PcaProjectionAdapter(n_components=pca2_dim), model_params=params
        )
    )

    cfg = GraphDroneConfig(
        router=SetRouterConfig(kind="bootstrap_full_only"),
        problem_type="classification",
        n_classes=n_classes,
    )
    gd = GraphDrone(cfg)
    gd.fit(X_tr, y_tr, expert_specs=specs)

    result = gd.predict(X_te, return_diagnostics=True)
    y_proba = result.predictions
    if y_proba.ndim == 1:
        y_proba = np.stack([1 - y_proba, y_proba], axis=1)
    y_pred = np.argmax(y_proba, axis=1)
    return y_pred, y_proba, result.diagnostics


def run_tabpfn(X_tr, y_tr, X_te, n_classes):
    """TabPFN baseline."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from tabpfn import TabPFNClassifier

    clf = TabPFNClassifier(n_estimators=8, device=device)
    clf.fit(X_tr, y_tr)
    y_proba = clf.predict_proba(X_te)
    y_pred = np.argmax(y_proba, axis=1)
    return y_pred, y_proba


# --------------------------------------------------------------------------- #
# Benchmark loop
# --------------------------------------------------------------------------- #
def run_benchmark(tasks=None, output_dir: Path = None, max_samples: int = 3000):
    from sklearn.model_selection import train_test_split

    if tasks is None:
        tasks = DEFAULT_TASKS
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "eval" / "tabarena_classification_bench"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for dataset_name, task_id in tasks:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}  (OpenML task {task_id})")
        print(f"{'='*60}")

        try:
            X, y, n_classes = load_openml_task(task_id, max_samples=max_samples)
        except Exception as e:
            print(f"  [SKIP] Failed to load: {e}")
            continue

        n_total = len(X)
        print(f"  N={n_total}, features={X.shape[1]}, n_classes={n_classes}")

        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            # Fallback: some rare classes have only 1 member, skip stratification
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        row = {"dataset": dataset_name, "task_id": task_id,
               "n_total": n_total, "n_features": X.shape[1], "n_classes": n_classes}

        # --- TabPFN ---
        try:
            t0 = time.time()
            y_pred_pfn, y_proba_pfn = run_tabpfn(X_tr, y_tr, X_te, n_classes)
            t_pfn = time.time() - t0
            m_pfn = compute_metrics(y_te, y_pred_pfn, y_proba_pfn, n_classes)
            row.update({f"tabpfn_{k}": v for k, v in m_pfn.items()})
            row["tabpfn_time"] = round(t_pfn, 2)
            print(f"  TabPFN      F1={m_pfn['f1_macro']:.4f}  ROC={m_pfn['auc_roc']:.4f}  PR={m_pfn['pr_auc']:.4f}  ({t_pfn:.1f}s)")
        except Exception as e:
            print(f"  TabPFN      FAILED: {e}")
            row.update({"tabpfn_f1_macro": None, "tabpfn_auc_roc": None, "tabpfn_pr_auc": None})

        # --- GraphDrone Baseline ---
        try:
            t0 = time.time()
            y_pred_base, y_proba_base, diag_base = run_graphdrone_baseline(X_tr, y_tr, X_te, y_te, n_classes)
            t_base = time.time() - t0
            m_base = compute_metrics(y_te, y_pred_base, y_proba_base, n_classes)
            row.update({f"gd_base_{k}": v for k, v in m_base.items()})
            row["gd_base_time"] = round(t_base, 2)
            row["gd_base_defer_rate"] = diag_base.get("mean_defer_prob", None)
            print(f"  GD-Baseline F1={m_base['f1_macro']:.4f}  ROC={m_base['auc_roc']:.4f}  PR={m_base['pr_auc']:.4f}  ({t_base:.1f}s)")
        except Exception as e:
            print(f"  GD-Baseline FAILED: {e}")
            import traceback
            traceback.print_exc()
            row.update({"gd_base_f1_macro": None, "gd_base_auc_roc": None, "gd_base_pr_auc": None})

        # --- GraphDrone HyperSetRouter ---
        try:
            t0 = time.time()
            y_pred_hyp, y_proba_hyp, diag_hyp = run_graphdrone_hyper(X_tr, y_tr, X_te, y_te, n_classes)
            t_hyp = time.time() - t0
            m_hyp = compute_metrics(y_te, y_pred_hyp, y_proba_hyp, n_classes)
            row.update({f"gd_hyper_{k}": v for k, v in m_hyp.items()})
            row["gd_hyper_time"] = round(t_hyp, 2)
            row["gd_hyper_defer_rate"] = diag_hyp.get("mean_defer_prob", None)
            print(f"  GD-Hyper    F1={m_hyp['f1_macro']:.4f}  ROC={m_hyp['auc_roc']:.4f}  PR={m_hyp['pr_auc']:.4f}  ({t_hyp:.1f}s)  defer={diag_hyp.get('mean_defer_prob', '?'):.3f}")
        except Exception as e:
            print(f"  GD-Hyper    FAILED: {e}")
            row.update({"gd_hyper_f1_macro": None, "gd_hyper_auc_roc": None, "gd_hyper_pr_auc": None})

        all_results.append(row)

        # Save incrementally
        df = pd.DataFrame(all_results)
        df.to_csv(output_dir / "results.csv", index=False)

    return pd.DataFrame(all_results)


# --------------------------------------------------------------------------- #
# Analysis: worst 3 datasets for GD-HyperSetRouter
# --------------------------------------------------------------------------- #
def analyze_worst_datasets(df: pd.DataFrame, output_dir: Path):
    """Detailed analysis of the 3 worst-performing datasets for GD-Hyper."""
    print("\n" + "="*70)
    print("WORST 3 DATASETS — GraphDrone-HyperSetRouter (by F1-macro)")
    print("="*70)

    # Compute composite score (mean rank across 3 metrics)
    metric_cols = ["gd_hyper_f1_macro", "gd_hyper_auc_roc", "gd_hyper_pr_auc"]
    valid = df.dropna(subset=metric_cols).copy()

    # Composite score: mean of the three metrics
    valid["gd_hyper_composite"] = valid[metric_cols].mean(axis=1).astype(float)

    worst3 = valid.nsmallest(3, "gd_hyper_composite")

    analysis_lines = []
    for _, row in worst3.iterrows():
        ds = row["dataset"]
        n_cls = int(row["n_classes"])
        n_tot = int(row["n_total"])
        n_feat = int(row["n_features"])

        def _f(v): return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else float("nan")

        hyp_f1  = _f(row.get("gd_hyper_f1_macro"))
        hyp_roc = _f(row.get("gd_hyper_auc_roc"))
        hyp_pr  = _f(row.get("gd_hyper_pr_auc"))
        base_f1 = _f(row.get("gd_base_f1_macro"))
        pfn_f1  = _f(row.get("tabpfn_f1_macro"))
        defer   = _f(row.get("gd_hyper_defer_rate"))

        delta_vs_base = hyp_f1 - base_f1 if not np.isnan(base_f1) else float("nan")
        delta_vs_pfn  = hyp_f1 - pfn_f1  if not np.isnan(pfn_f1)  else float("nan")

        block = f"""
Dataset: {ds}
  Config:  N={n_tot}, features={n_feat}, classes={n_cls}
  Metrics (GD-Hyper):    F1={hyp_f1:.4f}  ROC={hyp_roc:.4f}  PR={hyp_pr:.4f}
  Metrics (GD-Baseline): F1={base_f1:.4f}
  Metrics (TabPFN):      F1={pfn_f1:.4f}
  Delta vs Baseline:     {delta_vs_base:+.4f}
  Delta vs TabPFN:       {delta_vs_pfn:+.4f}
  Mean Defer Prob:       {defer:.4f}

  Failure Hypotheses:
"""
        # Hypothesis generation
        hypotheses = []
        if n_cls > 5:
            hypotheses.append("  • High class count — TabPFN softmax calibration degrades "
                              "beyond ~5 classes; specialist proba vectors become noisy, "
                              "reducing SNR for the router.")
        if n_feat > 50:
            hypotheses.append("  • High feature dimensionality — single FULL expert with "
                              "n_estimators=8 may underfit; view decomposition (subspace "
                              "specialists) needed to surface geometric structure.")
        if n_tot < 500:
            hypotheses.append("  • Small dataset — 80/20 split leaves <400 training samples; "
                              "router over-fits on validation fold, collapsing to full-only.")
        if not np.isnan(defer) and defer < 0.05:
            hypotheses.append("  • Near-zero defer rate — router effectively ignores "
                              "specialists (anchor dominates). TaskToken SNR signal may "
                              "be uninformative for this distribution.")
        if not np.isnan(delta_vs_base) and delta_vs_base < -0.02:
            hypotheses.append("  • HyperRouter hurts vs baseline — attention-gated routing "
                              "introduces variance; router training converges to a bad "
                              "local minimum when specialist proba variance is low.")
        if not hypotheses:
            hypotheses.append("  • Borderline case — composite score close to median; "
                              "additional folds needed to confirm underperformance.")

        block += "\n".join(hypotheses)
        block += "\n\n  Recommended fixes:\n"

        fixes = []
        if n_cls > 5:
            fixes.append("  → Use temperature-scaled proba from TabPFN; add per-class "
                         "entropy as an additional token feature.")
        if n_feat > 50:
            fixes.append("  → Add PCA/random-subspace view specialists to the expert "
                         "portfolio for structural diversity.")
        if not np.isnan(defer) and defer < 0.05:
            fixes.append("  → Increase patience / lower LR for router training; add "
                         "entropy-regularization term to prevent anchor collapse.")
        if not fixes:
            fixes.append("  → Run 5-fold CV to confirm; inspect per-class confusion matrix.")

        block += "\n".join(fixes)
        analysis_lines.append(block)
        print(block)

    report_path = output_dir / "worst3_analysis.txt"
    with open(report_path, "w") as f:
        f.write("WORST 3 DATASETS — GraphDrone HyperSetRouter\n")
        f.write("=" * 70 + "\n")
        f.write("\n".join(analysis_lines))
    print(f"\nAnalysis written to {report_path}")

    return worst3


# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #
def print_summary(df: pd.DataFrame, output_dir: Path):
    print("\n" + "="*70)
    print("FULL RESULTS SUMMARY")
    print("="*70)

    cols = ["dataset", "n_classes",
            "tabpfn_f1_macro", "gd_base_f1_macro", "gd_hyper_f1_macro",
            "tabpfn_auc_roc",  "gd_base_auc_roc",  "gd_hyper_auc_roc",
            "tabpfn_pr_auc",   "gd_base_pr_auc",   "gd_hyper_pr_auc",
            "gd_hyper_defer_rate"]
    show = df[[c for c in cols if c in df.columns]].copy()

    # Highlight winner per row
    for m in ["f1_macro", "auc_roc", "pr_auc"]:
        m_cols = [f"tabpfn_{m}", f"gd_base_{m}", f"gd_hyper_{m}"]
        present = [c for c in m_cols if c in show.columns]
        if present:
            show[f"best_{m}"] = show[present].idxmax(axis=1).str.replace(f"_{m}", "")

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(show.to_string(index=False))

    # Win-rate (use any two models that both have data)
    for m in ["f1_macro", "auc_roc", "pr_auc"]:
        hyp_col = f"gd_hyper_{m}"
        base_col = f"gd_base_{m}"
        pfn_col  = f"tabpfn_{m}"
        if hyp_col in df.columns and pfn_col in df.columns:
            valid_pfn = df[[hyp_col, pfn_col]].dropna()
            wins_vs_pfn = (valid_pfn[hyp_col] > valid_pfn[pfn_col]).sum()
            print(f"\n{m}: GD-Hyper beats TabPFN on {wins_vs_pfn}/{len(valid_pfn)} datasets")
        if hyp_col in df.columns and base_col in df.columns:
            valid_base = df[[hyp_col, base_col]].dropna()
            if len(valid_base) > 0:
                wins_vs_base = (valid_base[hyp_col] > valid_base[base_col]).sum()
                print(f"{m}: GD-Hyper beats GD-Baseline on {wins_vs_base}/{len(valid_base)} datasets")

    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(show.to_string(index=False))
    print(f"\nSummary written to {summary_path}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=3000,
                        help="Max samples per dataset (default 3000)")
    parser.add_argument("--datasets", nargs="+",
                        help="Subset of dataset names to run (default: all)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    tasks = DEFAULT_TASKS
    if args.datasets:
        name_set = set(args.datasets)
        tasks = [(n, t) for n, t in DEFAULT_TASKS if n in name_set]

    out_dir = Path(args.output_dir) if args.output_dir else None

    df = run_benchmark(tasks=tasks, output_dir=out_dir, max_samples=args.max_samples)

    if df.empty:
        print("No results — all datasets failed to load.")
        sys.exit(1)

    if out_dir is None:
        out_dir = Path(__file__).parent.parent / "eval" / "tabarena_classification_bench"

    print_summary(df, out_dir)
    analyze_worst_datasets(df, out_dir)

    print(f"\nResults CSV: {out_dir / 'results.csv'}")
