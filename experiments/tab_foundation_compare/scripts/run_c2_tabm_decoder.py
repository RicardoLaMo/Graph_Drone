from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.tab_foundation_compare.src.c2_decoder_experiment import run_c2_decoder_experiment
from experiments.tab_foundation_compare.src.repo_refs import REPO_REFERENCES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    artifacts_dir = root / "artifacts"
    reports_dir = root / "reports"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    result = run_c2_decoder_experiment(artifacts_dir, smoke=args.smoke)
    suffix = "__smoke" if args.smoke else ""
    metrics_path = artifacts_dir / f"c2_decoder_metrics{suffix}.csv"
    report_path = reports_dir / f"c2_decoder_report{suffix}.md"

    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "split",
                "rmse",
                "mae",
                "r2",
                "best_epoch",
                "duration_seconds",
                "gate_entropy",
                "gate_top1_mass",
                "head_prediction_std",
            ],
        )
        writer.writeheader()
        for item in result["results"]:
            for split in ["train", "val", "test"]:
                writer.writerow(
                    {
                        "model": item["model"],
                        "split": split,
                        "rmse": item[split]["rmse"],
                        "mae": item[split]["mae"],
                        "r2": item[split]["r2"],
                        "best_epoch": item["best_epoch"],
                        "duration_seconds": item["duration_seconds"],
                        "gate_entropy": item["diagnostics"].get(f"{split}_gate_entropy", ""),
                        "gate_top1_mass": item["diagnostics"].get(f"{split}_gate_top1_mass", ""),
                        "head_prediction_std": item["diagnostics"].get(
                            f"{split}_head_prediction_std", ""
                        ),
                    }
                )

    lines = [
        "# C2 Decoder Challenger",
        "",
        "Aligned California split, decoder-only comparison on top of a TabM-style ensemble backbone.",
        "",
        "## Repo References",
        "",
    ]
    for ref in REPO_REFERENCES:
        lines.append(f"- `{ref.model}`: RMSE `{ref.test_rmse:.4f}`")
    lines.extend(["", "## Results", ""])
    for item in result["results"]:
        lines.append(
            f"- `{item['model']}`: val RMSE `{item['val']['rmse']:.4f}`, "
            f"test RMSE `{item['test']['rmse']:.4f}`, "
            f"best epoch `{item['best_epoch']}`, duration `{item['duration_seconds']:.1f}s`"
        )
        if "test_gate_entropy" in item["diagnostics"]:
            lines.append(
                f"  gate entropy `{item['diagnostics']['test_gate_entropy']:.4f}`, "
                f"top-1 gate mass `{item['diagnostics']['test_gate_top1_mass']:.4f}`, "
                f"head prediction std `{item['diagnostics']['test_head_prediction_std']:.4f}`"
            )
    if len(result["results"]) == 2:
        delta = result["results"][0]["test"]["rmse"] - result["results"][1]["test"]["rmse"]
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                f"- Decoder gating delta on test RMSE: `{delta:.4f}` "
                "(positive means the gated decoder is better).",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n")
    print(report_path)
    print(metrics_path)


if __name__ == "__main__":
    main()
