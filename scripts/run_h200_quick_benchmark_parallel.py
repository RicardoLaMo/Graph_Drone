#!/usr/bin/env python3
"""
H200 Parallel Quick Benchmark - Extended Stress Testing Across Multiple GPUs

This script runs the quick benchmark (11 synthetic datasets) in parallel across
H200 GPUs 1-5 for validation before full TabArena. It demonstrates:
- Multi-GPU distribution
- Parallel task execution
- GPU load balancing
- H200 memory efficiency

Each GPU runs:
- 2-3 parallel quick benchmark instances
- 11 synthetic datasets each
- Aggregated metrics

Usage:
    python scripts/run_h200_quick_benchmark_parallel.py --gpus 1 2 3 4 5
"""

import os
import sys
import json
import argparse
import time
import logging
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, Manager
from dataclasses import dataclass

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from graphdrone_fit import GraphDrone, GraphDroneConfig, SetRouterConfig
from graphdrone_fit.config import SetRouterConfig

# Import validation scripts
import importlib.util
spec = importlib.util.spec_from_file_location("quick_benchmark", ROOT / "validation_scripts" / "quick_benchmark.py")
quick_bench_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quick_bench_module)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | GPU %(gpu_id)d | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QuickBenchmarkTask:
    gpu_id: int
    task_id: int
    baseline_name: str = "v1-width"
    candidate_name: str = "pc-moe-multiclass"
    status: str = "pending"
    result: dict = None
    error: str = None


def run_quick_benchmark_on_gpu(
    task: QuickBenchmarkTask,
    gpu_id: int,
) -> QuickBenchmarkTask:
    """Run quick benchmark on a specific GPU."""
    task.gpu_id = gpu_id

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        logger.info(f"Starting task {task.task_id}", extra={"gpu_id": gpu_id})

        # Setup models
        baseline_config = GraphDroneConfig(
            problem_type="classification",
            n_classes=3,
            router=SetRouterConfig(kind="contextual_transformer_router")
        )
        candidate_config = GraphDroneConfig(
            problem_type="classification",
            n_classes=3,
            router=SetRouterConfig(kind="contextual_transformer_router")
        )

        baseline_model = GraphDrone(baseline_config)
        candidate_model = GraphDrone(candidate_config)

        # Run quick benchmark
        orchestrator = QuickBenchmarkOrchestrator(
            baseline_model=baseline_model,
            candidate_model=candidate_model,
            baseline_name=task.baseline_name,
            candidate_name=task.candidate_name
        )

        start_time = time.time()
        report, total_time = orchestrator.run(quick_test=False)
        duration = time.time() - start_time

        # Collect results
        task.result = {
            "status": "completed",
            "total_time": total_time,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
        }
        task.status = "completed"

        logger.info(
            f"Task {task.task_id} completed in {duration:.1f}s",
            extra={"gpu_id": gpu_id}
        )

        return task

    except Exception as e:
        logger.error(
            f"Task {task.task_id} failed: {e}",
            extra={"gpu_id": gpu_id}
        )
        task.status = "failed"
        task.error = str(e)
        return task


def run_parallel_h200_benchmark(gpu_ids: list, n_tasks: int = 10):
    """Run multiple quick benchmarks in parallel across GPUs."""
    logger.info(f"Starting H200 parallel quick benchmark")
    logger.info(f"GPUs: {gpu_ids}")
    logger.info(f"Tasks: {n_tasks}")

    # Create tasks
    tasks = [
        QuickBenchmarkTask(
            gpu_id=gpu_ids[i % len(gpu_ids)],
            task_id=i
        )
        for i in range(n_tasks)
    ]

    # Run in parallel
    results_dir = ROOT / "eval" / "h200_quick_benchmark"
    results_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    completed = 0
    failed = 0

    # Process tasks with workers per GPU
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = {
            executor.submit(run_quick_benchmark_on_gpu, task, task.gpu_id): task
            for task in tasks
        }

        for future in as_completed(futures):
            task = futures[future]
            try:
                result_task = future.result()

                if result_task.status == "completed":
                    completed += 1
                    logger.info(f"✅ Task {result_task.task_id} completed")
                else:
                    failed += 1
                    logger.error(f"❌ Task {result_task.task_id} failed: {result_task.error}")

                # Save individual result
                result_file = results_dir / f"task_{result_task.task_id}.json"
                with open(result_file, "w") as f:
                    json.dump({
                        "task_id": result_task.task_id,
                        "gpu_id": result_task.gpu_id,
                        "status": result_task.status,
                        "result": result_task.result,
                        "error": result_task.error,
                    }, f, indent=2)

            except Exception as e:
                logger.error(f"Failed to process task: {e}")
                failed += 1

    elapsed = time.time() - start_time

    # Summary
    logger.info("="*70)
    logger.info("H200 PARALLEL QUICK BENCHMARK SUMMARY")
    logger.info("="*70)
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    logger.info(f"Completed: {completed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {100*completed/(completed+failed):.1f}%")
    logger.info(f"Results saved to: {results_dir}")
    logger.info("="*70)

    # Save summary
    summary_file = results_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "gpus": gpu_ids,
            "total_tasks": n_tasks,
            "completed": completed,
            "failed": failed,
            "total_time": elapsed,
            "success_rate": 100*completed/(completed+failed),
        }, f, indent=2)

    return completed, failed


def main():
    parser = argparse.ArgumentParser(
        description="H200 Parallel Quick Benchmark"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="GPU IDs to use",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=10,
        help="Number of parallel quick benchmark tasks",
    )

    args = parser.parse_args()

    print("="*70)
    print("H200 PARALLEL QUICK BENCHMARK - Validation Run")
    print("="*70)
    print(f"GPUs: {args.gpus}")
    print(f"Tasks: {args.tasks}")
    print(f"Expected runtime: {args.tasks * 2}s - {args.tasks * 3}s per GPU")
    print("="*70)
    print()

    completed, failed = run_parallel_h200_benchmark(args.gpus, args.tasks)

    if failed == 0:
        print("✅ All tasks completed successfully!")
        return 0
    else:
        print(f"⚠️ {failed} tasks failed")
        return 1


if __name__ == "__main__":
    exit(main())
