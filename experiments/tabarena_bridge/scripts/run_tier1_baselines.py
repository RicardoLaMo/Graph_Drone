from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tabarena_bridge.src.baselines import run_hgbr, run_tabpfn
from experiments.tabarena_bridge.src.datasets import build_openml_regression_split
from experiments.tabarena_bridge.src.manifest import load_manifest, select_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="houses")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "tabarena_bridge" / "reports" / "tier1_baselines",
    )
    return parser.parse_args()


def _dataset_entry(name: str) -> dict:
    manifest = load_manifest()
    rows = select_datasets(manifest, max_tier=1)
    for row in rows:
        if row["name"] == name:
            return row
    raise KeyError(f"Unknown Tier 1 dataset: {name}")


def _write_results(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    rows = payload["results"]
    with (output_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# Tier 1 Baselines — {payload['dataset']}",
        "",
        f"- task id: `{payload['task_id']}`",
        f"- split seed: `{payload['split_seed']}`",
        f"- train seed: `{payload['seed']}`",
        f"- rows: train `{payload['train_rows']}`, val `{payload['val_rows']}`, test `{payload['test_rows']}`",
        f"- features after preprocessing: `{payload['n_features']}`",
        "",
        "| Model | Val RMSE | Test RMSE | Val MAE | Test MAE | Notes |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['val_rmse']:.4f} | {row['test_rmse']:.4f} | "
            f"{row['val_mae']:.4f} | {row['test_mae']:.4f} | {row['notes']} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    entry = _dataset_entry(args.dataset)
    split = build_openml_regression_split(task_id=entry["openml_task_id"], split_seed=args.split_seed)

    max_train_samples = 512 if args.smoke else None

    results = []
    for baseline in [
        run_hgbr(split=split, seed=args.seed),
        run_tabpfn(
            split=split,
            seed=args.seed,
            n_estimators=1 if args.smoke else args.n_estimators,
            device=args.device,
            max_train_samples=max_train_samples,
        ),
    ]:
        results.append(
            {
                "model": baseline.model,
                "val_rmse": baseline.val_rmse,
                "test_rmse": baseline.test_rmse,
                "val_mae": baseline.val_mae,
                "test_mae": baseline.test_mae,
                "val_r2": baseline.val_r2,
                "test_r2": baseline.test_r2,
                "notes": baseline.notes,
            }
        )

    payload = {
        "dataset": entry["name"],
        "task_id": entry["openml_task_id"],
        "split_seed": args.split_seed,
        "seed": args.seed,
        "train_rows": split.train_rows,
        "val_rows": split.val_rows,
        "test_rows": split.test_rows,
        "n_features": split.n_features,
        "results": results,
    }
    output_dir = args.output / f"{entry['name']}__split{args.split_seed}__seed{args.seed}"
    _write_results(output_dir, payload)
    print(json.dumps({"dataset": entry["name"], "results": results}, indent=2))


if __name__ == "__main__":
    main()
