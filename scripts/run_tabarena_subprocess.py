#!/usr/bin/env python3
"""
TabArena Benchmark Runner using Subprocess (No CUDA_VISIBLE_DEVICES)

Uses subprocess.run() for clean process isolation without GPU device remapping.
Each task runs in a separate process with fresh Python environment.

Usage:
    python scripts/run_tabarena_subprocess.py
"""

import os
import sys
import json
import logging
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Setup path
ROOT = Path(__file__).parent.parent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(ROOT / "logs" / "tabarena_subprocess.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_single_task_subprocess(dataset_name: str, fold: int, task_id: int) -> dict:
    """Run a single task using subprocess (clean process isolation)."""

    script = f"""
import sys
sys.path.insert(0, '{ROOT / "src"}')

from tabarena.benchmark.experiment import Experiment, ExperimentBatchRunner
from graphdrone_fit.adapters.tabarena import GraphDroneTabArenaAdapter

expname = '{ROOT / "experiments" / "tabarena_subprocess"}'
task_metadata = {{
    'datasets': {{'name': '{dataset_name}', 'fold': {fold}}},
    'methods': ['GraphDrone_PC_MoE']
}}

exp_batch_runner = ExperimentBatchRunner(
    expname=expname,
    task_metadata=task_metadata,
)

methods = [
    Experiment(
        name="GraphDrone_PC_MoE_H200",
        method_cls=GraphDroneTabArenaAdapter,
        method_kwargs={{
            "n_estimators": 8,
            "router_kind": "contextual_transformer_router",
        }}
    ),
]

try:
    results = exp_batch_runner.run(
        datasets=['{dataset_name}'],
        folds=[{fold}],
        methods=methods,
        ignore_cache=False,
    )
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {{e}}")
    sys.exit(1)
"""

    start_time = time.time()

    try:
        # Run in subprocess WITHOUT setting CUDA_VISIBLE_DEVICES
        # Let PyTorch handle GPU selection naturally
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour per task
        )

        duration = time.time() - start_time

        if result.returncode == 0 and "SUCCESS" in result.stdout:
            logger.info(f"✅ Task {task_id}: {dataset_name} (fold {fold}) completed ({duration:.1f}s)")
            return {
                "task_id": task_id,
                "dataset": dataset_name,
                "fold": fold,
                "status": "completed",
                "duration": duration,
            }
        else:
            error_msg = result.stderr[-200:] if result.stderr else result.stdout[-200:]
            logger.error(f"❌ Task {task_id}: {dataset_name} (fold {fold}) failed: {error_msg}")
            return {
                "task_id": task_id,
                "dataset": dataset_name,
                "fold": fold,
                "status": "failed",
                "duration": duration,
                "error": error_msg,
            }

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        logger.error(f"❌ Task {task_id}: {dataset_name} (fold {fold}) timeout after {duration:.1f}s")
        return {
            "task_id": task_id,
            "dataset": dataset_name,
            "fold": fold,
            "status": "failed",
            "duration": duration,
            "error": "Timeout",
        }
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Task {task_id}: {dataset_name} (fold {fold}) error: {e}")
        return {
            "task_id": task_id,
            "dataset": dataset_name,
            "fold": fold,
            "status": "failed",
            "duration": duration,
            "error": str(e),
        }


def main():
    logger.info("="*80)
    logger.info("TABARENA SUBPROCESS BENCHMARK RUNNER")
    logger.info("="*80)

    # Datasets from TabArena
    datasets = [
        "APSFailure", "Amazon_employee_access", "Another-Dataset-on-used-Fiat-500",
        "Bank_Customer_Churn", "Bioresponse", "Diabetes130US", "GesturePhaseSegmentationProcessed",
        "hedge_fund_trading", "Housing", "Insurance_Company_Benchmark", "MiceProtein",
        "PhishingWebsites", "Polish_companies_bankruptcy", "Salary_Data", "seismic-bumps",
        "splice", "students_dropout_and_academic_success", "superconductivity",
        "taiwanese_bankruptcy_prediction", "website_phishing", "wine_quality",
        "adult", "breast_cancer_wisconsin", "car_evaluation", "contraceptive_method_choice",
        "Devnagari_Script_characters", "german_credit", "glass_identification", "heart_disease",
        "Hungarian_chickenpox", "ionosphere", "iris", "letter_recognition", "mfeat-factors",
        "Mice_Protein_Expression", "monks_problems", "mushroom", "online_shoppers_purchasing_intention",
        "ozone_level_8hr", "page_blocks", "passenger_satisfaction", "phoneme", "pollen",
        "postoperative_patient_data", "protein_fold_prediction", "qsar_fish_toxicity",
        "rice_cammeo_kinema", "road_safety", "sensorless_drive_diagnosis", "user_knowledge_modeling",
        "wine"
    ]

    folds = 3

    # Create tasks
    tasks = []
    task_id = 0
    for dataset in datasets:
        for fold in range(folds):
            tasks.append((dataset, fold, task_id))
            task_id += 1

    logger.info(f"Total tasks: {len(tasks)}")
    logger.info(f"Running tasks sequentially (subprocess isolation)")
    logger.info("="*80)
    logger.info("")

    # Create output directory
    (ROOT / "eval" / "tabarena_subprocess").mkdir(parents=True, exist_ok=True)

    # Run tasks
    start_time = time.time()
    results = []
    completed = 0
    failed = 0

    for dataset, fold, task_id in tasks:
        result = run_single_task_subprocess(dataset, fold, task_id)
        results.append(result)

        if result["status"] == "completed":
            completed += 1
        else:
            failed += 1

        progress = (completed + failed) / len(tasks) * 100
        logger.info(f"Progress: {completed + failed}/{len(tasks)} ({progress:.1f}%) | "
                   f"Completed: {completed}, Failed: {failed}")

    elapsed = time.time() - start_time

    # Summary
    logger.info("")
    logger.info("="*80)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*80)
    logger.info(f"Total time: {elapsed/3600:.2f} hours")
    logger.info(f"Completed: {completed}/{len(tasks)}")
    logger.info(f"Failed: {failed}/{len(tasks)}")
    logger.info(f"Success rate: {100*completed/len(tasks):.1f}%")
    logger.info("="*80)

    # Save results
    result_file = ROOT / "eval" / "tabarena_subprocess" / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_tasks": len(tasks),
            "completed": completed,
            "failed": failed,
            "success_rate": 100*completed/len(tasks),
            "total_time": elapsed,
            "results": results,
        }, f, indent=2)

    logger.info(f"Results saved to {result_file}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
