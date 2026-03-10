from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "experiments" / "tabarena_bridge" / "configs" / "p0_position_sources.json"
DEFAULT_OUT = ROOT / "experiments" / "tabarena_bridge" / "artifacts" / "p0_positioning"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _extract_p0_seed_result(path: Path, model_name: str) -> dict:
    payload = _load_json(path)
    row = next(row for row in payload["results"] if row["model"] == model_name)
    return {
        "seed": int(payload["seed"]),
        "split_seed": int(payload["split_seed"]),
        "val_rmse": float(row["val_rmse"]),
        "test_rmse": float(row["test_rmse"]),
    }


def _extract_tabpfn_multiseed(path: Path) -> list[dict]:
    payload = _load_json(path)
    rows = []
    for row in payload["runs"]:
        rows.append(
            {
                "seed": int(row["seed"]),
                "split_seed": 42,
                "val_rmse": float(row["val_rmse"]),
                "test_rmse": float(row["test_rmse"]),
            }
        )
    return rows


def _extract_foundation_single(path: Path, selector: dict) -> list[dict]:
    df = pd.read_csv(path)
    matched = df[(df["stage"] == selector["stage"]) & (df["model"] == selector["model"])]
    if matched.empty:
        raise KeyError(f"No row matched selector={selector} in {path}")
    row = matched.iloc[0]
    return [
        {
            "seed": None,
            "split_seed": None,
            "val_rmse": float(row["val_rmse"]) if not math.isnan(row["val_rmse"]) else None,
            "test_rmse": float(row["test_rmse"]),
        }
    ]


def _extract_scalar(value: float) -> list[dict]:
    return [{"seed": None, "split_seed": None, "val_rmse": None, "test_rmse": float(value)}]


def _load_method_runs(spec: dict) -> list[dict]:
    kind = spec["kind"]
    if kind == "p0_per_seed":
        return [_extract_p0_seed_result(Path(path), spec["model"]) for path in spec["paths"]]
    if kind == "tabpfn_multiseed":
        return _extract_tabpfn_multiseed(Path(spec["path"]))
    if kind == "foundation_csv_single":
        return _extract_foundation_single(Path(spec["path"]), spec["selector"])
    if kind == "scalar":
        return _extract_scalar(spec["value"])
    raise ValueError(f"Unsupported source kind: {kind}")


def _summarize_method(name: str, coverage: str, rows: list[dict]) -> dict:
    test_values = [row["test_rmse"] for row in rows]
    val_values = [row["val_rmse"] for row in rows if row["val_rmse"] is not None]
    return {
        "model": name,
        "coverage": coverage,
        "n_runs": len(rows),
        "mean_test_rmse": sum(test_values) / len(test_values),
        "std_test_rmse": float(pd.Series(test_values).std(ddof=0)) if len(test_values) > 1 else None,
        "mean_val_rmse": sum(val_values) / len(val_values) if val_values else None,
    }


def _normalized_scores(rows: list[dict]) -> list[dict]:
    best = min(row["mean_test_rmse"] for row in rows)
    worst = max(row["mean_test_rmse"] for row in rows)
    denom = worst - best
    for idx, row in enumerate(sorted(rows, key=lambda item: item["mean_test_rmse"]), start=1):
        row["rank"] = idx
        row["normalized_score"] = 1.0 if denom == 0 else (worst - row["mean_test_rmse"]) / denom
    return sorted(rows, key=lambda item: item["rank"])


def _paired_win_stats(target_name: str, method_runs: dict[str, list[dict]]) -> dict[str, dict]:
    target_runs = {row["seed"]: row for row in method_runs[target_name] if row["seed"] is not None}
    out: dict[str, dict] = {}
    for name, runs in method_runs.items():
        if name == target_name:
            continue
        comparable = []
        for row in runs:
            if row["seed"] is None or row["seed"] not in target_runs:
                continue
            comparable.append((row["seed"], row["test_rmse"], target_runs[row["seed"]]["test_rmse"]))
        if not comparable:
            out[name] = {"comparable_runs": 0, "target_wins": None, "mean_delta": None}
            continue
        target_wins = sum(1 for _, other, target in comparable if target < other)
        mean_delta = sum(other - target for _, other, target in comparable) / len(comparable)
        out[name] = {
            "comparable_runs": len(comparable),
            "target_wins": target_wins,
            "mean_delta": mean_delta,
        }
    return out


def _write_outputs(output_dir: Path, summary_rows: list[dict], paired: dict[str, dict], benchmark_name: str, target_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["rank", "model", "coverage", "n_runs", "mean_test_rmse", "std_test_rmse", "mean_val_rmse", "normalized_score"]
    with (output_dir / "leaderboard.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    payload = {
        "benchmark_name": benchmark_name,
        "target_model": target_name,
        "leaderboard": summary_rows,
        "paired_vs_target": paired,
    }
    (output_dir / "leaderboard.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        f"# P0 Positioning — {benchmark_name}",
        "",
        "This is a provisional TabArena-style leaderboard for the California anchor benchmark.",
        "It uses mean RMSE, normalized score, and paired win counts when split-wise runs exist.",
        "",
        f"- target model: `{target_name}`",
        "",
        "| Rank | Model | Coverage | Runs | Mean Test RMSE | Std | Normalized Score |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        std = "-" if row["std_test_rmse"] is None else f"{row['std_test_rmse']:.4f}"
        lines.append(
            f"| {row['rank']} | {row['model']} | {row['coverage']} | {row['n_runs']} | "
            f"{row['mean_test_rmse']:.4f} | {std} | {row['normalized_score']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"## Paired Wins vs `{target_name}`",
            "",
            "| Model | Comparable Runs | Target Wins | Mean Test Delta (positive favors target) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, row in paired.items():
        wins = "-" if row["target_wins"] is None else str(row["target_wins"])
        delta = "-" if row["mean_delta"] is None else f"{row['mean_delta']:.4f}"
        lines.append(f"| {name} | {row['comparable_runs']} | {wins} | {delta} |")
    (output_dir / "position_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    config = _load_json(args.config)
    method_runs = {spec["name"]: _load_method_runs(spec) for spec in config["methods"]}
    summary_rows = [
        _summarize_method(spec["name"], spec["coverage"], method_runs[spec["name"]])
        for spec in config["methods"]
    ]
    summary_rows = _normalized_scores(summary_rows)
    paired = _paired_win_stats(config["target_model"], method_runs)
    _write_outputs(args.output, summary_rows, paired, config["benchmark_name"], config["target_model"])
    print(json.dumps({"output_dir": str(args.output), "leaderboard_models": [row["model"] for row in summary_rows]}, indent=2))


if __name__ == "__main__":
    main()
