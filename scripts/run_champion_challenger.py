#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from graphdrone_fit.champion_challenger import (
    COMPLETED_STATUSES,
    build_dataset_summary,
    build_markdown_report,
    build_paired_task_table,
    collect_coverage_issues,
    evaluate_promotion,
    load_results_csv,
    write_comparison_artifacts,
)
from graphdrone_fit.claim_checks import build_claim_markdown, evaluate_claims, write_claim_artifacts
from graphdrone_fit.presets import available_graphdrone_presets
from graphdrone_fit.run_provenance import append_run_event, finalize_run_ledger, start_run_ledger


SCRIPT_BY_TASK = {
    "regression": ROOT / "scripts" / "run_geopoe_benchmark.py",
    "classification": ROOT / "scripts" / "run_smart_benchmark.py",
}


def _contract_tasks(selection: str) -> list[str]:
    if selection == "both":
        return ["regression", "classification"]
    return [selection]


def _contract_args(*, task: str, gate: str, datasets: list[str] | None, folds: list[int] | None) -> list[str]:
    if datasets:
        args = ["--datasets", *datasets]
        if folds:
            args.extend(["--folds", *[str(fold) for fold in folds]])
        return args
    if folds:
        return ["--tasks", task, "--folds", *[str(fold) for fold in folds]]
    if gate == "quick":
        quick_datasets = {
            "regression": ["california", "cpu_act"],
            "classification": ["pendigits"],
        }
        return ["--datasets", *quick_datasets[task], "--folds", "0"]
    if gate == "mini-full":
        return ["--tasks", task, "--folds", "0"]
    return ["--tasks", task, "--folds", "0", "1", "2"]


def _pythonpath_env() -> str:
    current = os.environ.get("PYTHONPATH", "")
    src_path = str(ROOT / "src")
    return src_path if not current else f"{src_path}{os.pathsep}{current}"


def _run_backend(
    *,
    label: str,
    preset: str,
    version: str,
    task: str,
    gate: str,
    methods: list[str],
    output_root: Path,
    datasets: list[str] | None,
    folds: list[int] | None,
    max_samples: int,
    heartbeat_seconds: float,
) -> Path:
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath_env()
    env["GRAPHDRONE_PRESET"] = preset
    env["GRAPHDRONE_VERSION_OVERRIDE"] = version

    run_dir = output_root / "raw" / label / task
    cache_dir = run_dir / "cache"
    output_dir = run_dir / "report"
    cmd = [
        sys.executable,
        str(SCRIPT_BY_TASK[task]),
        *_contract_args(task=task, gate=gate, datasets=datasets, folds=folds),
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(output_dir),
        "--max-samples",
        str(max_samples),
        "--methods",
        *methods,
    ]
    print(f"\n[{label}:{task}] {' '.join(cmd)}")
    process = subprocess.Popen(cmd, cwd=ROOT, env=env)
    ledger_path = start_run_ledger(
        run_dir=run_dir,
        root=ROOT,
        label=label,
        task=task,
        cmd=cmd,
        env=env,
        child_pid=process.pid,
    )
    append_run_event(run_dir=run_dir, child_pid=process.pid, event="started")
    started = time.monotonic()
    while True:
        try:
            returncode = process.wait(timeout=heartbeat_seconds)
            break
        except subprocess.TimeoutExpired:
            append_run_event(run_dir=run_dir, child_pid=process.pid, event="heartbeat")
    elapsed_seconds = time.monotonic() - started
    append_run_event(run_dir=run_dir, child_pid=process.pid, event="finished", returncode=returncode)
    finalize_run_ledger(ledger_path=ledger_path, elapsed_seconds=elapsed_seconds, returncode=returncode)
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)
    return output_dir / "results_granular.csv"


def _aggregate_reference(label: str, df: pd.DataFrame) -> pd.DataFrame:
    ok_df = df[df["status"].isin(COMPLETED_STATUSES)].copy()
    if ok_df.empty:
        return ok_df
    keep_cols = [
        col
        for col in ["rmse", "mae", "r2", "f1_macro", "log_loss", "auc_roc", "pr_auc", "elapsed"]
        if col in ok_df.columns
    ]
    ref = ok_df.groupby(["dataset", "task_type"], dropna=False)[keep_cols].mean().reset_index()
    ref["method"] = label
    return ref


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphDrone champion/challenger benchmark runner")
    parser.add_argument("--task", choices=["regression", "classification", "both"], default="both")
    parser.add_argument("--gate", choices=["quick", "mini-full", "full"], default="mini-full")
    parser.add_argument("--datasets", nargs="+", default=None, help="Override the contract with explicit dataset names")
    parser.add_argument("--folds", nargs="+", type=int, default=None, help="Override the contract with explicit folds")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--champion-preset", choices=available_graphdrone_presets(), default="v1_20_champion")
    parser.add_argument("--challenger-preset", choices=available_graphdrone_presets(), default="afc_candidate")
    parser.add_argument("--champion-version", default="champion-v1.20.0")
    parser.add_argument("--challenger-version", default="challenger-current")
    parser.add_argument("--with-tabpfn-anchor", action="store_true")
    parser.add_argument("--efficiency-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "eval" / "champion_challenger")
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    args = parser.parse_args()
    args.heartbeat_seconds = max(float(args.heartbeat_seconds), 1.0)

    tasks = _contract_tasks(args.task)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    champion_frames: list[pd.DataFrame] = []
    challenger_frames: list[pd.DataFrame] = []
    anchor_frames: list[pd.DataFrame] = []

    print("=" * 80)
    print("GRAPHDRONE CHAMPION / CHALLENGER")
    print("=" * 80)
    print(f"  Tasks            : {tasks}")
    print(f"  Gate             : {args.gate}")
    print(f"  Champion preset  : {args.champion_preset} ({args.champion_version})")
    print(f"  Challenger preset: {args.challenger_preset} ({args.challenger_version})")
    print(f"  TabPFN anchor    : {args.with_tabpfn_anchor}")
    print(f"  Efficiency only  : {args.efficiency_only}")
    print(f"  Output dir       : {args.output_dir}")
    print(f"  Heartbeat secs   : {args.heartbeat_seconds}")

    for task in tasks:
        champion_csv = _run_backend(
            label="champion",
            preset=args.champion_preset,
            version=args.champion_version,
            task=task,
            gate=args.gate,
            methods=["graphdrone"],
            output_root=args.output_dir,
            datasets=args.datasets,
            folds=args.folds,
            max_samples=args.max_samples,
            heartbeat_seconds=args.heartbeat_seconds,
        )
        challenger_csv = _run_backend(
            label="challenger",
            preset=args.challenger_preset,
            version=args.challenger_version,
            task=task,
            gate=args.gate,
            methods=["graphdrone"],
            output_root=args.output_dir,
            datasets=args.datasets,
            folds=args.folds,
            max_samples=args.max_samples,
            heartbeat_seconds=args.heartbeat_seconds,
        )
        champion_frames.append(load_results_csv(champion_csv, method_label="champion"))
        challenger_frames.append(load_results_csv(challenger_csv, method_label="challenger"))

        if args.with_tabpfn_anchor:
            anchor_csv = _run_backend(
                label="tabpfn_anchor",
                preset=args.champion_preset,
                version=f"anchor-{task}",
                task=task,
                gate=args.gate,
                methods=["tabpfn"],
                output_root=args.output_dir,
                datasets=args.datasets,
                folds=args.folds,
                max_samples=args.max_samples,
                heartbeat_seconds=args.heartbeat_seconds,
            )
            anchor_frames.append(load_results_csv(anchor_csv, method_label="tabpfn"))

    champion_df = pd.concat(champion_frames, ignore_index=True) if champion_frames else pd.DataFrame()
    challenger_df = pd.concat(challenger_frames, ignore_index=True) if challenger_frames else pd.DataFrame()
    paired_df = build_paired_task_table(champion_df, challenger_df)
    dataset_summary = build_dataset_summary(paired_df)
    coverage_issues = collect_coverage_issues(champion_df, challenger_df)
    decision = evaluate_promotion(
        paired_df,
        dataset_summary,
        efficiency_only=args.efficiency_only,
        coverage_issues=coverage_issues,
    )

    anchor_reference = None
    if anchor_frames:
        anchor_df = pd.concat(anchor_frames, ignore_index=True)
        anchor_reference = pd.concat(
            [
                _aggregate_reference("champion", champion_df),
                _aggregate_reference("challenger", challenger_df),
                _aggregate_reference("tabpfn", anchor_df),
            ],
            ignore_index=True,
        )

    report = build_markdown_report(
        decision=decision,
        dataset_summary=dataset_summary,
        paired_df=paired_df,
        champion_name=args.champion_version,
        challenger_name=args.challenger_version,
        anchor_reference=anchor_reference,
    )
    comparison_dir = args.output_dir / "comparison"
    write_comparison_artifacts(
        output_dir=comparison_dir,
        paired_df=paired_df,
        dataset_summary=dataset_summary,
        decision=decision,
        markdown_report=report,
        anchor_reference=anchor_reference,
    )
    claim_report = evaluate_claims(paired_df)
    write_claim_artifacts(
        output_dir=comparison_dir,
        claim_report=claim_report,
        markdown_report=build_claim_markdown(claim_report),
    )
    metadata = {
        "tasks": tasks,
        "gate": args.gate,
        "datasets": args.datasets,
        "folds": args.folds,
        "champion_preset": args.champion_preset,
        "champion_version": args.champion_version,
        "challenger_preset": args.challenger_preset,
        "challenger_version": args.challenger_version,
        "with_tabpfn_anchor": args.with_tabpfn_anchor,
        "efficiency_only": args.efficiency_only,
        "max_samples": args.max_samples,
        "heartbeat_seconds": args.heartbeat_seconds,
    }
    (comparison_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print("\nDecision:", decision["status"])
    for task_type, task_decision in decision["task_decisions"].items():
        if not task_decision["applicable"]:
            continue
        print(f"  {task_type}: {'PASS' if task_decision['pass'] else 'FAIL'}")
        for reason in task_decision["reasons"]:
            print(f"    - {reason}")
    if decision["coverage_issues"]:
        print("  coverage issues:")
        for issue in decision["coverage_issues"]:
            print(f"    - {issue}")
    print(f"\nArtifacts written to {comparison_dir}")


if __name__ == "__main__":
    main()
