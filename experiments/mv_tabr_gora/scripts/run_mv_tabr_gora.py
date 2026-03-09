"""
run_mv_tabr_gora.py — A0→A6 ablation experiment runner.

Usage
-----
  # Fast smoke test (~30s):
  cd /Volumes/MacMini/Projects/Graph_Drone
  python3 experiments/mv_tabr_gora/scripts/run_mv_tabr_gora.py --smoke

  # Full run (A0-A6):
  python3 experiments/mv_tabr_gora/scripts/run_mv_tabr_gora.py

  # Single ablation:
  python3 experiments/mv_tabr_gora/scripts/run_mv_tabr_gora.py --ablation A3

Ablation ladder
---------------
  A0: global kNN (FULL view only), uniform attention, label in KV
  A1: per-view kNN, uniform routing, label in KV
  A2: per-view kNN, sigma2_v routing + J-temperature
  A3: A2 + T(z_i^v − z_j^v) direction encoding in value
  A4: A3 + Q(q_i^v, q_j^v) quality-pair encoding in value
  A5: A4 + learned routing MLP
  A6: A5 + CrossViewMixer + β-gate (isolation/interaction blend)

Reference baselines (from aligned CA protocol, reports/):
  TabR         0.3829  (champion)
  TabM         0.4290
  HGBR         0.4430
  GoRA v2 G2   0.4546
  L2_stack_hgbr 0.4292  (non-parametric, from validation)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mv_tabr_gora.src.data import build_mv_data_bundle
from experiments.mv_tabr_gora.src.model import MVTabrGoraModel, ModelConfig, ablation_config
from experiments.mv_tabr_gora.src.train import TrainConfig, get_device, train_mv_tabr_gora


REFERENCE_BASELINES = {
    "TabR (champion)":   0.3829,
    "TabM":              0.4290,
    "L2_stack_hgbr":     0.4292,
    "HGBR":              0.4430,
    "GoRA v2 G2":        0.4546,
    "GoRA v5 A0":        0.4738,
}

ALL_ABLATIONS = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A4f", "A5f", "A6f"]

ABLATION_DESCRIPTIONS = {
    "A0": "Global kNN (FULL view), uniform attention, label in KV",
    "A1": "Per-view kNN, uniform routing, label in KV",
    "A2": "Per-view kNN, sigma2_v routing + J-temperature",
    "A3": "A2 + T(z_i^v − z_j^v) direction encoding in value",
    "A4": "A3 + Q(q_i^v, q_j^v) quality-pair encoding in value",
    "A5": "A4 + learned routing MLP",
    "A6": "A5 + CrossViewMixer + β-gate (isolation/interaction blend)",
    # Fixed (no Q in value; Q hurt in run 1)
    "A4f": "A3 + learned routing (no Q in value) [fixed]",
    "A5f": "A3 + CrossViewMixer + β-gate (no Q in value) [fixed]",
    "A6f": "A3 + learned routing + CrossViewMixer + β-gate (no Q) [fixed]",
}


def param_count(model: MVTabrGoraModel) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_ablation(
    name: str,
    bundle,
    train_cfg: TrainConfig,
    device,
    d_model: int = 64,
    K: int = 24,
    verbose: bool = True,
) -> dict:
    """Train one ablation and return result dict."""
    print(f"\n{'='*60}")
    print(f"  {name}: {ABLATION_DESCRIPTIONS[name]}")
    print(f"{'='*60}")

    cfg = ablation_config(
        name,
        view_dims=bundle.view_dims,
        view_names=bundle.view_names,
        K=bundle.K,
        d_model=d_model,
    )
    model = MVTabrGoraModel(cfg)
    n_params = param_count(model)
    print(f"  Parameters: {n_params:,}")

    result = train_mv_tabr_gora(model, bundle, train_cfg, device, verbose=verbose)
    result["model"] = name
    result["description"] = ABLATION_DESCRIPTIONS[name]
    result["n_params"] = n_params

    test_rmse = result["test"]["rmse"]
    val_rmse = result["val"]["rmse"]
    print(f"  RESULT  val_rmse={val_rmse:.4f}  test_rmse={test_rmse:.4f}  "
          f"best_epoch={result['best_epoch']}")

    return result


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 75)
    print("  MV-TabR-GoRA Ablation Summary")
    print("=" * 75)
    print(f"  {'Model':<16} {'Test RMSE':>10} {'vs A0':>8} {'Val RMSE':>10}  Description")
    print("  " + "-" * 70)

    # Compute delta vs A0
    a0_rmse = next((r["test"]["rmse"] for r in results if r["model"] == "A0"), None)

    for r in results:
        test_rmse = r["test"]["rmse"]
        val_rmse = r["val"]["rmse"]
        delta = f"{a0_rmse - test_rmse:+.4f}" if a0_rmse is not None else "  —   "
        desc = r["description"][:42]
        print(f"  {r['model']:<16} {test_rmse:>10.4f} {delta:>8}  {val_rmse:>10.4f}  {desc}")

    print("\n  Reference baselines:")
    for name, rmse in REFERENCE_BASELINES.items():
        print(f"  {'':16} {rmse:>10.4f} {'':>8}  {'':>10}  {name}")
    print("=" * 75)


def write_report(
    results: list[dict],
    output_dir: Path,
    bundle_info: dict,
    total_seconds: float,
    smoke: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = "mv_tabr_gora_results__smoke.json" if smoke else "mv_tabr_gora_results.json"
    payload = {
        "bundle_info": bundle_info,
        "reference_baselines": REFERENCE_BASELINES,
        "ablation_descriptions": ABLATION_DESCRIPTIONS,
        "results": results,
        "total_seconds": round(total_seconds, 1),
    }
    (output_dir / fname).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n  Report written → {output_dir / fname}")


def main():
    parser = argparse.ArgumentParser(description="MV-TabR-GoRA ablation experiment")
    parser.add_argument("--smoke", action="store_true", help="Fast smoke subset")
    parser.add_argument("--ablation", nargs="+", choices=ALL_ABLATIONS,
                        default=ALL_ABLATIONS, help="Which ablations to run")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--K", type=int, default=24, help="Neighbours per view")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "experiments/mv_tabr_gora/reports")
    args = parser.parse_args()

    smoke = args.smoke
    ablations_to_run = args.ablation

    # ---- Build dataset --------------------------------------------------
    print("Building MV data bundle ...")
    t0 = time.time()
    bundle = build_mv_data_bundle(
        K=args.K,
        smoke=smoke,
        smoke_train=500,
        smoke_val=200,
        smoke_test=200,
    )
    build_time = time.time() - t0
    print(f"  Built in {build_time:.1f}s")
    print(f"  N_train={len(bundle.train_idx)}  N_val={len(bundle.val_idx)}  "
          f"N_test={len(bundle.test_idx)}  K={bundle.K}  V={len(bundle.view_names)}")
    print(f"  Views: {bundle.view_names}")
    print(f"  sigma2_v range: [{bundle.sigma2_v.min():.3f}, {bundle.sigma2_v.max():.3f}]")
    print(f"  mean_J range: [{bundle.mean_J.min():.4f}, {bundle.mean_J.max():.4f}]")

    bundle_info = {
        "n_train": int(len(bundle.train_idx)),
        "n_val": int(len(bundle.val_idx)),
        "n_test": int(len(bundle.test_idx)),
        "K": bundle.K,
        "V": len(bundle.view_names),
        "view_names": bundle.view_names,
        "view_dims": bundle.view_dims,
        "d_model": args.d_model,
        "smoke": smoke,
    }

    # ---- Training config ------------------------------------------------
    if smoke:
        train_cfg = TrainConfig(
            batch_size=64, max_epochs=8, patience=4, lr=3e-4, seed=0
        )
    else:
        train_cfg = TrainConfig(
            batch_size=512, max_epochs=150, patience=30, lr=3e-4, seed=0
        )

    device = get_device()
    print(f"  Device: {device}")

    # ---- Run ablations --------------------------------------------------
    results = []
    t_total_start = time.time()

    for name in ablations_to_run:
        result = run_ablation(
            name, bundle, train_cfg, device,
            d_model=args.d_model, K=args.K, verbose=True,
        )
        results.append(result)

    total_seconds = time.time() - t_total_start

    # ---- Summary + report -----------------------------------------------
    print_summary(results)
    write_report(results, args.output, bundle_info, total_seconds, smoke)


if __name__ == "__main__":
    main()
