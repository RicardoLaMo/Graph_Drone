#!/usr/bin/env python3
"""
Full TabArena Benchmark Runner for Multiclass Refactor Validation

This script runs the PC-MoE multiclass refactor across all 51 TabArena datasets
with 3 folds to generate complete ELO metrics and evidence for PR validation.

Usage:
    conda run -n h200_tabpfn python scripts/run_full_tabarena_benchmark.py [OPTIONS]

Options:
    --datasets N        Run on N datasets (default: 51 for full run)
    --folds N           Run N folds per dataset (default: 3 for full validation)
    --quick             Quick test on 5 datasets, 1 fold (overrides defaults)
    --output-dir DIR    Output directory for results (default: eval/tabarena_full)
    --device cuda/cpu   Device for router (default: cuda)
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

# Add src and external/tabarena to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "tabarena" / "tabarena"))

try:
    from tabarena.benchmark.experiment import Experiment, ExperimentBatchRunner
    from tabarena.nips2025_utils.end_to_end import EndToEnd
    from tabarena.nips2025_utils.tabarena_context import TabArenaContext
    from tabarena.website.website_format import format_leaderboard
    from graphdrone_fit.adapters.tabarena import GraphDroneTabArenaAdapter
except ImportError as e:
    print(f"ERROR: TabArena not installed. Run in h200_tabpfn environment:")
    print("  conda run -n h200_tabpfn python scripts/run_full_tabarena_benchmark.py")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Full TabArena Benchmark for Multiclass Refactor"
    )
    parser.add_argument(
        "--datasets",
        type=int,
        default=51,
        help="Number of datasets to run (default: 51 for full)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Number of folds per dataset (default: 3 for validation)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test on 5 datasets, 1 fold",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval/tabarena_full",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for router training",
    )

    args = parser.parse_args()

    # Override with quick test if specified
    if args.quick:
        args.datasets = 5
        args.folds = 1
        args.output_dir = "eval/tabarena_quick"
        print("Running QUICK TEST: 5 datasets, 1 fold")
    else:
        print(f"Running FULL BENCHMARK: {args.datasets} datasets, {args.folds} folds")
        print(f"Expected runtime: 4-6 hours on 6x A100 GPUs")

    # Setup
    expname = str(ROOT / "experiments" / "tabarena_multiclass_refactor")
    eval_dir = Path(ROOT / args.output_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()
    print(f"\n{'='*70}")
    print(f"Starting TabArena Benchmark: {timestamp}")
    print(f"{'='*70}")
    print(f"Output directory: {eval_dir}")
    print(f"Datasets: {args.datasets}")
    print(f"Folds per dataset: {args.folds}")
    print(f"Device: {args.device}")
    print(f"{'='*70}\n")

    # Load TabArena context
    print("Loading TabArena context...")
    tabarena_context = TabArenaContext()
    task_metadata = tabarena_context.task_metadata
    all_names = list(task_metadata["name"].unique())

    print(f"Available datasets: {len(all_names)}")

    # Select datasets
    datasets = all_names[:args.datasets]
    folds = list(range(args.folds))

    print(f"Selected {len(datasets)} datasets")
    print(f"Folds: {folds}\n")

    # Define method
    methods = [
        Experiment(
            name="GraphDrone_PC_MoE_Multiclass",
            method_cls=GraphDroneTabArenaAdapter,
            method_kwargs={
                "n_estimators": 8,
                "router_kind": "noise_gate_router",
                "device": args.device,
            }
        ),
    ]

    # Run benchmark
    print("Starting benchmark run...")
    exp_batch_runner = ExperimentBatchRunner(
        expname=expname,
        task_metadata=task_metadata,
    )

    results_lst: list[dict[str, Any]] = exp_batch_runner.run(
        datasets=datasets,
        folds=folds,
        methods=methods,
        ignore_cache=False,
    )

    print(f"\nCompleted {len(results_lst)} benchmark runs")

    # Compute results
    print("\nComputing results and ELO metrics...")
    end_to_end = EndToEnd.from_raw(
        results_lst=results_lst,
        task_metadata=task_metadata,
        cache=False,
        cache_raw=False,
    )
    end_to_end_results = end_to_end.to_results()

    # Generate leaderboard
    print("\nGenerating leaderboard...")
    leaderboard: pd.DataFrame = end_to_end_results.compare_on_tabarena(
        output_dir=eval_dir,
        only_valid_tasks=True,
        use_model_results=True,
        new_result_prefix="PC_MoE_",
    )

    # Print results
    leaderboard_website = format_leaderboard(df_leaderboard=leaderboard)
    print("\n" + "="*70)
    print("LEADERBOARD RESULTS")
    print("="*70)
    print(leaderboard_website.to_markdown(index=False))

    # Extract ELO
    if "PC_MoE_elo" in leaderboard.columns:
        elos = leaderboard["PC_MoE_elo"].dropna()
        if len(elos) > 0:
            mean_elo = elos.mean()
            print(f"\n{'='*70}")
            print(f"Mean ELO: {mean_elo:.1f}")
            print(f"{'='*70}")

    # Save leaderboard to CSV
    leaderboard_path = eval_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path)
    print(f"\nLeaderboard saved: {leaderboard_path}")

    # Create metrics summary
    metrics_summary = {
        "experiment": "PC-MoE Multiclass Refactor",
        "timestamp": timestamp,
        "datasets_tested": len(datasets),
        "folds_per_dataset": args.folds,
        "total_runs": len(results_lst),
        "device": args.device,
        "n_estimators": 8,
        "router_kind": "noise_gate_router",
        "leaderboard_path": str(leaderboard_path),
        "results_dir": str(eval_dir),
    }

    # If ELO available, add it
    if "PC_MoE_elo" in leaderboard.columns:
        elos = leaderboard["PC_MoE_elo"].dropna()
        if len(elos) > 0:
            metrics_summary["mean_elo"] = float(elos.mean())
            metrics_summary["std_elo"] = float(elos.std())
            metrics_summary["n_valid_tasks"] = len(elos)

    metrics_path = eval_dir / "metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Metrics summary saved: {metrics_path}")

    print(f"\n{'='*70}")
    print("✅ Benchmark completed successfully")
    print(f"Results: {eval_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
