from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _mean(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return float("nan")
    values = pd.to_numeric(df[column], errors="coerce")
    if values.notna().sum() == 0:
        return float("nan")
    return float(values.mean())


def _fraction(df: pd.DataFrame, mask: pd.Series) -> float:
    if len(df) == 0:
        return float("nan")
    return float(mask.fillna(False).mean())


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


def _router_mask(df: pd.DataFrame, needle: str) -> pd.Series:
    if "router_kind_challenger" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["router_kind_challenger"].fillna("").astype(str).str.contains(needle, regex=False)


def _bottom_line_summary(rows: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if "rmse_rel_improvement" in rows.columns:
        summary["mean_rmse_rel_improvement"] = _mean(rows, "rmse_rel_improvement")
    if "r2_delta" in rows.columns:
        summary["mean_r2_delta"] = _mean(rows, "r2_delta")
    if "f1_delta" in rows.columns:
        summary["mean_f1_delta"] = _mean(rows, "f1_delta")
    if "log_loss_rel_improvement" in rows.columns:
        summary["mean_log_loss_rel_improvement"] = _mean(rows, "log_loss_rel_improvement")
    return summary


def _coupling_status(summary: dict[str, Any]) -> str:
    reg = summary.get("mean_rmse_rel_improvement")
    clf_f1 = summary.get("mean_f1_delta")
    clf_ll = summary.get("mean_log_loss_rel_improvement")
    reg_positive = reg is not None and not np.isnan(reg) and reg > 0
    clf_positive = (
        (clf_f1 is not None and not np.isnan(clf_f1) and clf_f1 > 0)
        or (clf_ll is not None and not np.isnan(clf_ll) and clf_ll > 0)
    )
    reg_negative = reg is not None and not np.isnan(reg) and reg < 0
    clf_negative = (
        (clf_f1 is not None and not np.isnan(clf_f1) and clf_f1 < 0)
        and (clf_ll is not None and not np.isnan(clf_ll) and clf_ll < 0)
    )
    if reg_positive or clf_positive:
        return "translating"
    if reg_negative or clf_negative:
        return "not_translating"
    return "flat"


def _claim_result(
    *,
    name: str,
    applicable: bool,
    component_status: str,
    reasons: list[str],
    activated_rows: pd.DataFrame | None = None,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active = activated_rows if activated_rows is not None else pd.DataFrame()
    bottom_line = _bottom_line_summary(active) if not active.empty else {}
    return {
        "name": name,
        "applicable": applicable,
        "component_status": component_status,
        "integration_status": (
            "inactive"
            if component_status in {"inactive", "insufficient_data"}
            else _coupling_status(bottom_line)
        ),
        "activated_task_count": int(len(active)),
        "reasons": reasons,
        "summary": {
            **(summary or {}),
            **bottom_line,
        },
    }


def _evaluate_legitimacy_claim(paired_df: pd.DataFrame) -> dict[str, Any]:
    applicable = "exit_frac_challenger" in paired_df.columns
    if not applicable or paired_df.empty:
        return _claim_result(
            name="legitimacy_gate",
            applicable=False,
            component_status="insufficient_data",
            reasons=["no legitimacy-gate diagnostics in paired results"],
        )

    active_mask = pd.to_numeric(paired_df["exit_frac_challenger"], errors="coerce").fillna(0.0) > 0.0
    active_rows = paired_df[active_mask]
    mean_exit_frac = _mean(paired_df, "exit_frac_challenger")
    mean_latency = _mean(active_rows if not active_rows.empty else paired_df, "latency_improvement")
    if active_rows.empty:
        return _claim_result(
            name="legitimacy_gate",
            applicable=True,
            component_status="inactive",
            reasons=["challenger never early-exited any evaluated task"],
            summary={
                "mean_exit_frac": mean_exit_frac,
                "activated_task_fraction": _fraction(paired_df, active_mask),
                "mean_latency_improvement_active": mean_latency,
            },
        )

    reasons = [
        f"mean_exit_frac={mean_exit_frac:.6f}",
        f"mean_latency_improvement_active={mean_latency:.6f}",
    ]
    component_status = "supported" if mean_latency > 0 else "contradicted"
    if component_status == "contradicted":
        reasons.append("gate fires, but active-task latency does not improve")
    return _claim_result(
        name="legitimacy_gate",
        applicable=True,
        component_status=component_status,
        reasons=reasons,
        activated_rows=active_rows,
        summary={
            "mean_exit_frac": mean_exit_frac,
            "activated_task_fraction": _fraction(paired_df, active_mask),
            "mean_latency_improvement_active": mean_latency,
        },
    )


def _evaluate_rotor_claim(paired_df: pd.DataFrame) -> dict[str, Any]:
    applicable = _router_mask(paired_df, "rotor").any()
    if not applicable or paired_df.empty:
        return _claim_result(
            name="rotor_alignment",
            applicable=False,
            component_status="insufficient_data",
            reasons=["no rotor challenger rows in paired results"],
        )

    mean_specialists = _mean(paired_df, "n_specialists_challenger")
    mean_gain = _mean(paired_df, "alignment_cosine_gain_challenger")
    active_mask = pd.to_numeric(
        paired_df.get("alignment_cosine_gain_challenger", pd.Series(np.nan, index=paired_df.index)),
        errors="coerce",
    ).fillna(0.0) > 0.0
    active_rows = paired_df[active_mask]

    if not np.isnan(mean_specialists) and mean_specialists <= 0:
        return _claim_result(
            name="rotor_alignment",
            applicable=True,
            component_status="inactive",
            reasons=["challenger has no specialists, so rotor cannot affect routing"],
            summary={
                "mean_n_specialists": mean_specialists,
                "mean_alignment_cosine_gain": mean_gain,
                "activated_task_fraction": _fraction(paired_df, active_mask),
            },
        )

    if np.isnan(mean_gain):
        return _claim_result(
            name="rotor_alignment",
            applicable=True,
            component_status="insufficient_data",
            reasons=["rotor challenger rows are present, but alignment gain diagnostics are missing"],
            summary={
                "mean_n_specialists": mean_specialists,
                "mean_alignment_cosine_gain": mean_gain,
            },
        )

    reasons = [
        f"mean_n_specialists={mean_specialists:.6f}" if not np.isnan(mean_specialists) else "mean_n_specialists=nan",
        f"mean_alignment_cosine_gain={mean_gain:.6f}",
    ]
    component_status = "supported" if mean_gain > 0 else "contradicted"
    if component_status == "contradicted":
        reasons.append("rotor does not improve anchor-specialist cosine alignment")
    return _claim_result(
        name="rotor_alignment",
        applicable=True,
        component_status=component_status,
        reasons=reasons,
        activated_rows=active_rows,
        summary={
            "mean_n_specialists": mean_specialists,
            "mean_alignment_cosine_gain": mean_gain,
            "activated_task_fraction": _fraction(paired_df, active_mask),
        },
    )


def _evaluate_ot_gate_claim(paired_df: pd.DataFrame) -> dict[str, Any]:
    applicable = _router_mask(paired_df, "ot").any()
    if not applicable or paired_df.empty:
        return _claim_result(
            name="ot_noise_gate",
            applicable=False,
            component_status="insufficient_data",
            reasons=["no OT-gate challenger rows in paired results"],
        )

    mean_ot_cost = _mean(paired_df, "mean_ot_cost_challenger")
    mean_validity = _mean(paired_df, "mean_specialist_validity_challenger")
    closed_frac = _mean(paired_df, "closed_specialist_frac_challenger")
    active_mask = pd.to_numeric(
        paired_df.get("closed_specialist_frac_challenger", pd.Series(np.nan, index=paired_df.index)),
        errors="coerce",
    ).fillna(0.0) > 0.0
    active_rows = paired_df[active_mask]

    if np.isnan(mean_ot_cost) or np.isnan(mean_validity):
        return _claim_result(
            name="ot_noise_gate",
            applicable=True,
            component_status="insufficient_data",
            reasons=["OT challenger rows are present, but OT diagnostics are missing"],
            summary={
                "mean_ot_cost": mean_ot_cost,
                "mean_specialist_validity": mean_validity,
                "mean_closed_specialist_frac": closed_frac,
            },
        )

    reasons = [
        f"mean_ot_cost={mean_ot_cost:.6f}",
        f"mean_specialist_validity={mean_validity:.6f}",
        f"mean_closed_specialist_frac={closed_frac:.6f}" if not np.isnan(closed_frac) else "mean_closed_specialist_frac=nan",
    ]
    if mean_validity >= 0.99 and (np.isnan(closed_frac) or closed_frac <= 0.0):
        component_status = "inactive"
        reasons.append("OT validity stays open, so the gate is not changing routing")
    else:
        component_status = "supported"
    return _claim_result(
        name="ot_noise_gate",
        applicable=True,
        component_status=component_status,
        reasons=reasons,
        activated_rows=active_rows,
        summary={
            "mean_ot_cost": mean_ot_cost,
            "mean_specialist_validity": mean_validity,
            "mean_closed_specialist_frac": closed_frac,
            "activated_task_fraction": _fraction(paired_df, active_mask),
        },
    )


def evaluate_claims(paired_df: pd.DataFrame) -> dict[str, Any]:
    claims = {
        "legitimacy_gate": _evaluate_legitimacy_claim(paired_df),
        "rotor_alignment": _evaluate_rotor_claim(paired_df),
        "ot_noise_gate": _evaluate_ot_gate_claim(paired_df),
    }
    applicable = [claim for claim in claims.values() if claim["applicable"]]
    return {
        "status": "complete" if applicable else "insufficient_data",
        "claims": claims,
    }


def build_claim_markdown(claim_report: dict[str, Any]) -> str:
    lines = [
        "# Component Claim Checks",
        "",
    ]
    for claim in claim_report["claims"].values():
        if not claim["applicable"]:
            continue
        lines.extend(
            [
                f"## {claim['name']}",
                "",
                f"- Component: `{claim['component_status']}`",
                f"- Integration: `{claim['integration_status']}`",
                f"- Activated tasks: `{claim['activated_task_count']}`",
            ]
        )
        lines.extend([f"- {reason}" for reason in claim["reasons"]])
        if claim["summary"]:
            lines.extend(["", "```text"])
            for key, value in claim["summary"].items():
                if isinstance(value, float) and not np.isnan(value):
                    lines.append(f"{key}: {value:.6f}")
                else:
                    lines.append(f"{key}: {value}")
            lines.extend(["```", ""])
        else:
            lines.append("")
    if len(lines) == 2:
        lines.append("No applicable claim checks.")
    return "\n".join(lines)


def write_claim_artifacts(
    *,
    output_dir: str | Path,
    claim_report: dict[str, Any],
    markdown_report: str,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "claim_report.json").write_text(json.dumps(_as_json_ready(claim_report), indent=2))
    (out_dir / "claim_report.md").write_text(markdown_report)
