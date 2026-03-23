from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


COMPLETED_STATUSES = frozenset({"ok", "cached"})
PAIR_KEYS = ["dataset", "fold", "task_type"]
REGRESSION_REQUIRED_COLUMNS = ("rmse", "mae", "r2", "elapsed")
CLASSIFICATION_REQUIRED_COLUMNS = ("f1_macro", "log_loss", "elapsed")
OPTIONAL_DELTA_COLUMNS = (
    "auc_roc",
    "pr_auc",
    "defer",
    "exit_frac",
    "legitimacy_score_mean",
    "n_experts",
    "n_specialists",
    "mean_ot_cost",
    "mean_specialist_validity",
    "closed_specialist_frac",
    "mean_specialist_mass",
    "mean_anchor_attention_weight",
    "alignment_aux_loss",
    "alignment_cosine_pre",
    "alignment_cosine_post",
    "alignment_cosine_gain",
    "non_anchor_attention_entropy",
)


def load_results_csv(path: str | Path, *, method_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["method"] = method_label
    return df


def _completed(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"].isin(COMPLETED_STATUSES)].copy()


def _task_key_tuples(df: pd.DataFrame) -> set[tuple[Any, ...]]:
    return {
        tuple(row)
        for row in df[PAIR_KEYS].itertuples(index=False, name=None)
    }


def collect_coverage_issues(champion_df: pd.DataFrame, challenger_df: pd.DataFrame) -> list[str]:
    issues: list[str] = []

    for label, df in [("champion", champion_df), ("challenger", challenger_df)]:
        failed = df[df["status"] == "fail"][PAIR_KEYS]
        if not failed.empty:
            preview = ", ".join(
                f"{row.dataset}/fold{row.fold}/{row.task_type}"
                for row in failed.head(5).itertuples(index=False)
            )
            suffix = " ..." if len(failed) > 5 else ""
            issues.append(f"{label} has {len(failed)} failed task(s): {preview}{suffix}")

    champion_ok = _task_key_tuples(_completed(champion_df))
    challenger_ok = _task_key_tuples(_completed(challenger_df))
    missing_in_champion = sorted(challenger_ok - champion_ok)
    missing_in_challenger = sorted(champion_ok - challenger_ok)
    if missing_in_champion:
        issues.append(f"champion is missing {len(missing_in_champion)} completed task(s) present in challenger")
    if missing_in_challenger:
        issues.append(f"challenger is missing {len(missing_in_challenger)} completed task(s) present in champion")
    return issues


def build_paired_task_table(champion_df: pd.DataFrame, challenger_df: pd.DataFrame) -> pd.DataFrame:
    champion_ok = _completed(champion_df)
    challenger_ok = _completed(challenger_df)
    merged = champion_ok.merge(
        challenger_ok,
        on=PAIR_KEYS,
        how="inner",
        suffixes=("_champion", "_challenger"),
    )
    if merged.empty:
        return merged

    eps = 1e-12
    if {"rmse_champion", "rmse_challenger"}.issubset(merged.columns):
        merged["rmse_delta"] = merged["rmse_challenger"] - merged["rmse_champion"]
        merged["rmse_rel_improvement"] = (
            merged["rmse_champion"] - merged["rmse_challenger"]
        ) / merged["rmse_champion"].abs().clip(lower=eps)
    if {"mae_champion", "mae_challenger"}.issubset(merged.columns):
        merged["mae_delta"] = merged["mae_challenger"] - merged["mae_champion"]
        merged["mae_rel_improvement"] = (
            merged["mae_champion"] - merged["mae_challenger"]
        ) / merged["mae_champion"].abs().clip(lower=eps)
    if {"r2_champion", "r2_challenger"}.issubset(merged.columns):
        merged["r2_delta"] = merged["r2_challenger"] - merged["r2_champion"]
    if {"f1_macro_champion", "f1_macro_challenger"}.issubset(merged.columns):
        merged["f1_delta"] = merged["f1_macro_challenger"] - merged["f1_macro_champion"]
    if {"log_loss_champion", "log_loss_challenger"}.issubset(merged.columns):
        merged["log_loss_delta"] = merged["log_loss_challenger"] - merged["log_loss_champion"]
        merged["log_loss_rel_improvement"] = (
            merged["log_loss_champion"] - merged["log_loss_challenger"]
        ) / merged["log_loss_champion"].abs().clip(lower=eps)
    if {"elapsed_champion", "elapsed_challenger"}.issubset(merged.columns):
        merged["latency_improvement"] = (
            merged["elapsed_champion"] - merged["elapsed_challenger"]
        ) / merged["elapsed_champion"].clip(lower=eps)

    for base_name in OPTIONAL_DELTA_COLUMNS:
        champion_col = f"{base_name}_champion"
        challenger_col = f"{base_name}_challenger"
        if champion_col in merged.columns and challenger_col in merged.columns:
            merged[f"{base_name}_delta"] = merged[challenger_col] - merged[champion_col]

    dynamic_attention_bases = set()
    for col in merged.columns:
        if not col.endswith("_champion"):
            continue
        base_name = col[: -len("_champion")]
        challenger_col = f"{base_name}_challenger"
        if challenger_col not in merged.columns:
            continue
        if base_name.startswith("mean_attention_"):
            dynamic_attention_bases.add(base_name)

    for base_name in sorted(dynamic_attention_bases):
        merged[f"{base_name}_delta"] = merged[f"{base_name}_challenger"] - merged[f"{base_name}_champion"]

    return merged


def build_dataset_summary(paired_df: pd.DataFrame) -> pd.DataFrame:
    if paired_df.empty:
        return paired_df.copy()

    exclude = {
        "status_champion",
        "status_challenger",
        "method_champion",
        "method_challenger",
        "router_kind_champion",
        "router_kind_challenger",
    }
    numeric_cols = [
        col
        for col in paired_df.columns
        if col not in PAIR_KEYS and col not in exclude and pd.api.types.is_numeric_dtype(paired_df[col])
    ]
    return (
        paired_df.groupby(["dataset", "task_type"], dropna=False)[numeric_cols]
        .mean()
        .reset_index()
        .sort_values(["task_type", "dataset"])
        .reset_index(drop=True)
    )


def _nan_to_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)) and np.isnan(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def _as_json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _as_json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_as_json_ready(v) for v in obj]
    return _nan_to_none(obj)


def _regression_decision(
    paired_df: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    *,
    efficiency_only: bool,
) -> dict[str, Any]:
    rows = paired_df[paired_df["task_type"] == "regression"]
    ds_rows = dataset_summary[dataset_summary["task_type"] == "regression"]
    if rows.empty:
        return {"applicable": False, "pass": True, "reasons": ["no regression rows"]}
    missing = [col for col in ("rmse_rel_improvement", "r2_delta") if col not in rows.columns]
    if missing:
        return {
            "applicable": True,
            "pass": False,
            "reasons": [f"missing regression comparison columns: {', '.join(missing)}"],
        }

    mean_rmse_improvement = float(rows["rmse_rel_improvement"].mean())
    mean_r2_delta = float(rows["r2_delta"].mean())
    worst_dataset_rmse = float(ds_rows["rmse_rel_improvement"].min()) if not ds_rows.empty else float("nan")
    severe_regression_frac = float((rows["rmse_rel_improvement"] < -0.05).mean())
    mean_latency_improvement = float(rows["latency_improvement"].mean()) if "latency_improvement" in rows else float("nan")

    checks: list[tuple[str, bool, float]] = []
    if efficiency_only:
        checks.extend(
            [
                ("mean_rmse_noninferior", mean_rmse_improvement >= -0.005, mean_rmse_improvement),
                ("mean_latency_improvement", mean_latency_improvement >= 0.10, mean_latency_improvement),
            ]
        )
    else:
        checks.extend(
            [
                ("mean_rmse_improvement", mean_rmse_improvement >= 0.005, mean_rmse_improvement),
                ("mean_r2_guardrail", mean_r2_delta >= -0.002, mean_r2_delta),
                ("worst_dataset_rmse_guardrail", worst_dataset_rmse >= -0.03, worst_dataset_rmse),
                ("severe_task_fraction_guardrail", severe_regression_frac <= 0.20, severe_regression_frac),
            ]
        )

    return {
        "applicable": True,
        "pass": all(item[1] for item in checks),
        "reasons": [
            f"{name}={'PASS' if ok else 'FAIL'} ({value:.6f})"
            for name, ok, value in checks
        ],
        "summary": {
            "mean_rmse_rel_improvement": mean_rmse_improvement,
            "mean_r2_delta": mean_r2_delta,
            "worst_dataset_rmse_rel_improvement": worst_dataset_rmse,
            "severe_task_fraction": severe_regression_frac,
            "mean_latency_improvement": mean_latency_improvement,
        },
    }


def _classification_decision(
    paired_df: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    *,
    efficiency_only: bool,
) -> dict[str, Any]:
    rows = paired_df[paired_df["task_type"] == "classification"]
    ds_rows = dataset_summary[dataset_summary["task_type"] == "classification"]
    if rows.empty:
        return {"applicable": False, "pass": True, "reasons": ["no classification rows"]}
    missing = [col for col in ("f1_delta", "log_loss_delta", "log_loss_rel_improvement") if col not in rows.columns]
    if missing:
        return {
            "applicable": True,
            "pass": False,
            "reasons": [f"missing classification comparison columns: {', '.join(missing)}"],
        }

    mean_f1_delta = float(rows["f1_delta"].mean())
    mean_log_loss_delta = float(rows["log_loss_delta"].mean())
    mean_log_loss_rel_improvement = float(rows["log_loss_rel_improvement"].mean())
    worst_dataset_f1 = float(ds_rows["f1_delta"].min()) if not ds_rows.empty else float("nan")
    worst_dataset_log_loss = float(ds_rows["log_loss_delta"].max()) if not ds_rows.empty else float("nan")
    mean_latency_improvement = float(rows["latency_improvement"].mean()) if "latency_improvement" in rows else float("nan")

    checks: list[tuple[str, bool, float]] = []
    if efficiency_only:
        checks.extend(
            [
                ("mean_f1_noninferior", mean_f1_delta >= -0.002, mean_f1_delta),
                ("mean_log_loss_noninferior", mean_log_loss_delta <= 0.01, mean_log_loss_delta),
                ("mean_latency_improvement", mean_latency_improvement >= 0.10, mean_latency_improvement),
            ]
        )
    else:
        checks.extend(
            [
                (
                    "headline_improvement",
                    (mean_f1_delta >= 0.002) or (mean_log_loss_rel_improvement >= 0.01),
                    max(mean_f1_delta, mean_log_loss_rel_improvement),
                ),
                ("mean_log_loss_guardrail", mean_log_loss_delta <= 0.0, mean_log_loss_delta),
                ("worst_dataset_f1_guardrail", worst_dataset_f1 >= -0.01, worst_dataset_f1),
                ("worst_dataset_log_loss_guardrail", worst_dataset_log_loss <= 0.03, worst_dataset_log_loss),
            ]
        )

    return {
        "applicable": True,
        "pass": all(item[1] for item in checks),
        "reasons": [
            f"{name}={'PASS' if ok else 'FAIL'} ({value:.6f})"
            for name, ok, value in checks
        ],
        "summary": {
            "mean_f1_delta": mean_f1_delta,
            "mean_log_loss_delta": mean_log_loss_delta,
            "mean_log_loss_rel_improvement": mean_log_loss_rel_improvement,
            "worst_dataset_f1_delta": worst_dataset_f1,
            "worst_dataset_log_loss_delta": worst_dataset_log_loss,
            "mean_latency_improvement": mean_latency_improvement,
        },
    }


def evaluate_promotion(
    paired_df: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    *,
    efficiency_only: bool = False,
    coverage_issues: list[str] | None = None,
) -> dict[str, Any]:
    issues = coverage_issues or []
    regression = _regression_decision(paired_df, dataset_summary, efficiency_only=efficiency_only)
    classification = _classification_decision(paired_df, dataset_summary, efficiency_only=efficiency_only)
    task_decisions = {
        "regression": regression,
        "classification": classification,
    }
    applicable = [decision for decision in task_decisions.values() if decision["applicable"]]
    overall_pass = not issues and all(decision["pass"] for decision in applicable)
    overall_status = "promote" if overall_pass else ("hold" if applicable else "insufficient_data")
    return {
        "status": overall_status,
        "pass": overall_pass,
        "efficiency_only": efficiency_only,
        "coverage_issues": issues,
        "task_decisions": task_decisions,
        "paired_task_count": int(len(paired_df)),
    }


def _format_table(df: pd.DataFrame, columns: list[str], *, sort_by: list[str]) -> str:
    if df.empty:
        return "(no rows)"
    table = df.sort_values(sort_by)[columns].copy()
    return table.to_string(index=False, float_format=lambda value: f"{value:.4f}")


def build_markdown_report(
    *,
    decision: dict[str, Any],
    dataset_summary: pd.DataFrame,
    paired_df: pd.DataFrame,
    champion_name: str,
    challenger_name: str,
    anchor_reference: pd.DataFrame | None = None,
) -> str:
    lines = [
        "# Champion vs Challenger",
        "",
        f"- Champion: `{champion_name}`",
        f"- Challenger: `{challenger_name}`",
        f"- Decision: `{decision['status']}`",
        f"- Paired tasks: `{decision['paired_task_count']}`",
        "",
    ]
    if decision["coverage_issues"]:
        lines.extend(["## Coverage Issues", ""])
        lines.extend([f"- {issue}" for issue in decision["coverage_issues"]])
        lines.append("")

    for task_type, heading, cols in [
        (
            "regression",
            "Regression",
            ["dataset", "rmse_rel_improvement", "mae_rel_improvement", "r2_delta", "latency_improvement"],
        ),
        (
            "classification",
            "Classification",
            ["dataset", "f1_delta", "log_loss_delta", "log_loss_rel_improvement", "latency_improvement"],
        ),
    ]:
        task_decision = decision["task_decisions"][task_type]
        if not task_decision["applicable"]:
            continue
        lines.extend([f"## {heading}", ""])
        lines.extend([f"- {reason}" for reason in task_decision["reasons"]])
        lines.extend(["", "```text"])
        task_summary = dataset_summary[dataset_summary["task_type"] == task_type]
        present_cols = [col for col in cols if col in task_summary.columns]
        lines.append(_format_table(task_summary, present_cols, sort_by=["dataset"]))
        lines.extend(["```", ""])

    if anchor_reference is not None and not anchor_reference.empty:
        lines.extend(["## TabPFN Anchor", "", "```text"])
        lines.append(
            _format_table(
                anchor_reference,
                [col for col in anchor_reference.columns if col != "task_type"],
                sort_by=["task_type", "dataset"],
            )
        )
        lines.extend(["```", ""])

    return "\n".join(lines)


def write_comparison_artifacts(
    *,
    output_dir: str | Path,
    paired_df: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    decision: dict[str, Any],
    markdown_report: str,
    anchor_reference: pd.DataFrame | None = None,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paired_df.to_csv(out_dir / "paired_task_deltas.csv", index=False)
    dataset_summary.to_csv(out_dir / "paired_dataset_summary.csv", index=False)
    if anchor_reference is not None and not anchor_reference.empty:
        anchor_reference.to_csv(out_dir / "tabpfn_anchor_summary.csv", index=False)
    (out_dir / "promotion_decision.json").write_text(json.dumps(_as_json_ready(decision), indent=2))
    (out_dir / "promotion_report.md").write_text(markdown_report)
