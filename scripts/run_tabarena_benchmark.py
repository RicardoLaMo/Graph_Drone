import os
import sys
import pandas as pd
from pathlib import Path
from typing import Any

# Add src and external/tabarena/tabarena to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "tabarena" / "tabarena"))

from tabarena.benchmark.experiment import Experiment, ExperimentBatchRunner
from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena.website.website_format import format_leaderboard
from graphdrone_fit.adapters.tabarena import GraphDroneTabArenaAdapter

if __name__ == '__main__':
    expname = str(Path(__file__).parent.parent / "experiments" / "tabarena_benchmark")
    eval_dir = Path(__file__).parent.parent / "eval" / "tabarena_benchmark"
    ignore_cache = False

    tabarena_context = TabArenaContext()
    task_metadata = tabarena_context.task_metadata

    # Benchmark Portfolio: 5 diverse datasets
    # Names from task_metadata: 'blood-transfusion-service-center', 'credit-g', 'diabetes', 'ionosphere', 'phoneme'
    # Wait, phoneme might not be in the 51. Let's check available names.
    all_names = list(task_metadata["name"].unique())
    print(f"Available datasets: {len(all_names)}")
    
    # Selection of 5 from the 51
    datasets = ["blood-transfusion-service-center", "credit-g", "diabetes", "ionosphere", "tic-tac-toe"]
    # Filter to only those present
    datasets = [d for d in datasets if d in all_names]
    
    folds = [0, 1] # Running 2 folds per dataset for the lab verification

    methods = [
        # Baseline: Single Global TabPFN (GraphDrone with 1 expert)
        # Note: We simulate TabPFN baseline by passing a single 'FULL' spec if we want to be exact,
        # but here we'll just compare against the official TabArena TabPFN results if possible.
        # For now, let's just run GraphDrone.
        Experiment(
            name="GraphDrone_v2026.03_Consolidated",
            method_cls=GraphDroneTabArenaAdapter,
            method_kwargs={
                "n_estimators": 8,
                "router_kind": "noise_gate_router"
            }
        ),
    ]

    exp_batch_runner = ExperimentBatchRunner(expname=expname, task_metadata=task_metadata)

    print(f"Starting TabArena Benchmark on {len(datasets)} datasets, {len(folds)} folds...")
    results_lst: list[dict[str, Any]] = exp_batch_runner.run(
        datasets=datasets,
        folds=folds,
        methods=methods,
        ignore_cache=ignore_cache,
    )

    # Compute results and compare against TabArena baselines
    end_to_end = EndToEnd.from_raw(results_lst=results_lst, task_metadata=task_metadata, cache=False, cache_raw=False)
    end_to_end_results = end_to_end.to_results()

    print("\n--- GRAPHDRONE VS TABARENA BASELINES ---")
    leaderboard: pd.DataFrame = end_to_end_results.compare_on_tabarena(
        output_dir=eval_dir,
        only_valid_tasks=True,
        use_model_results=True,
        new_result_prefix="GD_",
    )
    leaderboard_website = format_leaderboard(df_leaderboard=leaderboard)
    print(leaderboard_website.to_markdown(index=False))
