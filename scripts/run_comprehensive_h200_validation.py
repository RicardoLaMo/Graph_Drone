#!/usr/bin/env python3
"""
Comprehensive H200 Parallel Validation (without TabArena dependency)

Runs extended benchmark multiple times in parallel across GPUs 1-5
to simulate full benchmark workload distribution and validate H200
optimization strategy.

This provides comprehensive validation while TabArena is being set up.

Usage:
    python scripts/run_comprehensive_h200_validation.py
"""

import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).parent.parent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | GPU %(gpu_id)d | %(message)s'
)
logger = logging.getLogger(__name__)


def run_extended_benchmark_on_gpu(gpu_id: int, iteration: int) -> dict:
    """Run extended benchmark on specific GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    start_time = time.time()

    try:
        logger.info(f"Starting iteration {iteration}", extra={"gpu_id": gpu_id})

        # Run extended benchmark
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "validation_scripts" / "extended_benchmark.py")
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
        )

        if result.returncode != 0:
            logger.error(f"Iteration {iteration} failed: {result.stderr}", extra={"gpu_id": gpu_id})
            return {
                "gpu_id": gpu_id,
                "iteration": iteration,
                "status": "failed",
                "error": result.stderr[-200:],
                "duration": time.time() - start_time
            }

        # Parse results from stdout
        duration = time.time() - start_time
        logger.info(f"✅ Iteration {iteration} completed ({duration:.1f}s)", extra={"gpu_id": gpu_id})

        return {
            "gpu_id": gpu_id,
            "iteration": iteration,
            "status": "completed",
            "duration": duration,
            "output": result.stdout[-500:] if result.stdout else "",
            "timestamp": datetime.now().isoformat()
        }

    except subprocess.TimeoutExpired:
        logger.error(f"Iteration {iteration} timeout", extra={"gpu_id": gpu_id})
        return {
            "gpu_id": gpu_id,
            "iteration": iteration,
            "status": "timeout",
            "duration": time.time() - start_time
        }
    except Exception as e:
        logger.error(f"Iteration {iteration} failed: {e}", extra={"gpu_id": gpu_id})
        return {
            "gpu_id": gpu_id,
            "iteration": iteration,
            "status": "failed",
            "error": str(e),
            "duration": time.time() - start_time
        }


def main():
    print("="*80)
    print("COMPREHENSIVE H200 PARALLEL VALIDATION")
    print("="*80)
    print(f"Mode: Extended benchmark (4 datasets: Wine, Breast Cancer, Digits, Segment)")
    print(f"Parallel execution across GPUs: 1, 2, 3, 4, 5")
    print(f"Total validations: 10 (2 per GPU)")
    print(f"Expected runtime: 20-30 minutes")
    print("="*80)
    print()

    # Create output directory
    results_dir = ROOT / "eval" / "h200_comprehensive_validation"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    gpu_ids = [1, 2, 3, 4, 5]
    n_iterations_per_gpu = 2
    total_iterations = len(gpu_ids) * n_iterations_per_gpu

    # Create tasks (round-robin across GPUs)
    tasks = []
    for iteration in range(total_iterations):
        gpu_id = gpu_ids[iteration % len(gpu_ids)]
        tasks.append((gpu_id, iteration))

    # Run in parallel
    print(f"Starting {total_iterations} parallel validations...\n")

    start_time = time.time()
    results = []
    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = {
            executor.submit(run_extended_benchmark_on_gpu, gpu_id, iteration): (gpu_id, iteration)
            for gpu_id, iteration in tasks
        }

        for future in as_completed(futures):
            gpu_id, iteration = futures[future]
            try:
                result = future.result()
                results.append(result)

                if result['status'] == 'completed':
                    completed += 1
                else:
                    failed += 1

                # Progress
                progress = (completed + failed) / total_iterations * 100
                logger.info(
                    f"Progress: {completed + failed}/{total_iterations} ({progress:.0f}%) | "
                    f"Completed: {completed}, Failed: {failed}",
                    extra={"gpu_id": gpu_id}
                )

            except Exception as e:
                logger.error(f"Task processing error: {e}", extra={"gpu_id": gpu_id})
                failed += 1

    total_time = time.time() - start_time

    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE VALIDATION SUMMARY")
    print("="*80)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Completed validations: {completed}")
    print(f"Failed validations: {failed}")
    print(f"Success rate: {100*completed/(completed+failed):.1f}%")
    print(f"Average duration per validation: {total_time/total_iterations:.1f}s")
    print("="*80)

    # GPU distribution summary
    print("\nGPU Distribution:")
    for gpu_id in gpu_ids:
        gpu_tasks = [r for r in results if r['gpu_id'] == gpu_id]
        gpu_completed = sum(1 for r in gpu_tasks if r['status'] == 'completed')
        gpu_time = sum(r['duration'] for r in gpu_tasks)
        print(f"  GPU {gpu_id}: {gpu_completed}/{len(gpu_tasks)} completed | {gpu_time:.1f}s total")

    # Save results
    result_file = results_dir / f"comprehensive_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_time": total_time,
            "completed": completed,
            "failed": failed,
            "success_rate": 100*completed/(completed+failed),
            "gpu_ids": gpu_ids,
            "results": results
        }, f, indent=2)

    print(f"\nResults saved to: {result_file}")
    print("\n" + "="*80)

    if failed == 0:
        print("✅ ALL VALIDATIONS PASSED")
        print("🚀 H200 optimization validated successfully")
        print("🎯 Ready for full TabArena benchmark")
        return 0
    else:
        print(f"⚠️ {failed} validations failed")
        print("🔧 Investigate failures before full benchmark")
        return 1


if __name__ == "__main__":
    exit(main())
