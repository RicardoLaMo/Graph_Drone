from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mv_tabr_gora.src.data import build_cohort_residual_bundle, build_mv_data_bundle
from experiments.mv_tabr_gora.src.model import MVTabrGoraModel, ablation_config
from experiments.mv_tabr_gora.src.train import TrainConfig, get_device, train_mv_tabr_gora


def param_count(model: MVTabrGoraModel) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_one(label: str, bundle, train_cfg: TrainConfig, device, d_model: int = 64, verbose: bool = True) -> dict:
    cfg = ablation_config("A6f", view_dims=bundle.view_dims, view_names=bundle.view_names, K=bundle.K, d_model=d_model)
    model = MVTabrGoraModel(cfg)
    result = train_mv_tabr_gora(model, bundle, train_cfg, device, verbose=verbose)
    result["model"] = label
    result["n_params"] = param_count(model)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="A6f vs G0 cohort-residual comparison")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--K", type=int, default=24)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["raw", "g0a"],
        choices=["raw", "g0", "g0a", "g0c"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "experiments/mv_tabr_gora/reports/cohort_residual",
    )
    args = parser.parse_args()

    train_cfg = (
        TrainConfig(batch_size=64, max_epochs=8, patience=4, lr=3e-4, seed=args.seed)
        if args.smoke
        else TrainConfig(batch_size=512, max_epochs=150, patience=30, lr=3e-4, seed=args.seed)
    )
    device = get_device()
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    builders = {
        "raw": lambda: build_mv_data_bundle(
            K=args.K,
            smoke=args.smoke,
            smoke_train=500,
            smoke_val=200,
            smoke_test=200,
            seed=args.seed,
        ),
        "g0": lambda: build_cohort_residual_bundle(
            K=args.K,
            smoke=args.smoke,
            smoke_train=500,
            smoke_val=200,
            smoke_test=200,
            seed=args.seed,
            append_seg_mean_to_geo=True,
            normalize_by_residual_std=False,
        ),
        "g0a": lambda: build_cohort_residual_bundle(
            K=args.K,
            smoke=args.smoke,
            smoke_train=500,
            smoke_val=200,
            smoke_test=200,
            seed=args.seed,
            append_seg_mean_to_geo=False,   # G0a: only residual target; kNN unchanged
            normalize_by_residual_std=False,
        ),
        "g0c": lambda: build_cohort_residual_bundle(
            K=args.K,
            smoke=args.smoke,
            smoke_train=500,
            smoke_val=200,
            smoke_test=200,
            seed=args.seed,
            append_seg_mean_to_geo=False,   # G0c: residual target + proper scale calibration
            normalize_by_residual_std=True,
        ),
    }
    labels = {
        "raw": "A6f_raw",
        "g0":  "G0_cohort_residual",
        "g0a": "G0a_residual_only",
        "g0c": "G0c_residual_rescaled",
    }

    results = []
    t0 = time.time()
    for variant in args.variants:
        print(f"\n{'=' * 60}\n  {labels[variant]}\n{'=' * 60}")
        bundle = builders[variant]()
        result = run_one(labels[variant], bundle, train_cfg, device, d_model=args.d_model, verbose=True)
        results.append(result)
        print(
            f"  RESULT  val_rmse={result['val']['rmse']:.4f} "
            f"test_rmse={result['test']['rmse']:.4f} best_epoch={result['best_epoch']}"
        )

    payload = {
        "smoke": args.smoke,
        "K": args.K,
        "d_model": args.d_model,
        "seed": args.seed,
        "variants": args.variants,
        "results": results,
        "total_seconds": round(time.time() - t0, 1),
        "note": "TrainConfig already uses Huber by default; G1 is the existing baseline loss.",
    }
    out_name = "cohort_residual_results__smoke.json" if args.smoke else "cohort_residual_results.json"
    (output_dir / out_name).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nReport written to {output_dir / out_name}")


if __name__ == "__main__":
    main()
