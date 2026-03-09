from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKTREE_ROOT))

from experiments.california_geo_segmentation.src.data import build_geo_data_bundle
from experiments.california_geo_segmentation.src.eval import evaluate_models, results_to_frame
from experiments.california_geo_segmentation.src.segmentation import build_default_segmentations


def write_report(output_dir: Path, metrics_md: str, best_line: str, seg_note: str) -> None:
    report = f"""# California Geo Segmentation Report

## Question
Can latitude/longitude-derived pseudo-neighborhood segments recover useful latent
community signal beyond raw continuous GEO features?

## Summary
{best_line}

## Metrics
{metrics_md}

## Interpretation
- This experiment keeps the California split and preprocessing aligned with the v5/A6f line.
- Baseline `B1_HGBR_raw` already includes continuous latitude/longitude in the raw feature vector.
- Segment target priors are train-only. Validation/test rows receive statistics from train-fitted segments.
- `H3_all_geo_priors_shuffled` is a leakage control built from shuffled train targets only.
- Positive signal here justifies pushing geo segmentation into retrieval candidate pools or routing priors.
- Flat or negative signal suggests lat/lon continuous features already capture most of what these simple segmenters can recover.

## Segmentation note
{seg_note}
"""
    (output_dir / "report.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="California geo segmentation validation")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKTREE_ROOT / "experiments/california_geo_segmentation/reports",
    )
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir.parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_geo_data_bundle(smoke=args.smoke)
    segmentations = build_default_segmentations(bundle.raw_geo, bundle.y, bundle.train_idx)
    rng = np.random.default_rng(42)
    y_for_shuffled_stats = bundle.y.copy()
    y_for_shuffled_stats[bundle.train_idx] = bundle.y[bundle.train_idx][
        rng.permutation(len(bundle.train_idx))
    ]
    shuffled_segmentations = build_default_segmentations(
        bundle.raw_geo,
        y_for_shuffled_stats,
        bundle.train_idx,
    )
    results = evaluate_models(
        X=bundle.X,
        y=bundle.y,
        train_idx=bundle.train_idx,
        test_idx=bundle.test_idx,
        segmentations=segmentations,
        shuffled_segmentations=shuffled_segmentations,
    )

    df = results_to_frame(results)
    df.to_csv(artifacts_dir / ("metrics__smoke.csv" if args.smoke else "metrics.csv"), index=False)

    import pandas as pd

    summary_df = pd.DataFrame(
        [row for seg in segmentations.values() for row in seg.summary_rows]
    )
    summary_df.to_csv(
        artifacts_dir / ("segmentation_summary__smoke.csv" if args.smoke else "segmentation_summary.csv"),
        index=False,
    )

    best = df.iloc[0]
    baseline = df[df["model"] == "B1_HGBR_raw"].iloc[0]
    delta = baseline["rmse"] - best["rmse"]
    if best["model"] == "B1_HGBR_raw":
        best_line = "No geo segmentation variant beat the raw HGBR baseline."
    else:
        best_line = (
            f"Best model was `{best['model']}` at RMSE `{best['rmse']:.4f}`, "
            f"improving over raw HGBR by `{delta:+.4f}`."
        )

    metrics_md = df.to_markdown(index=False)
    seg_note = (
        "Schemes tested: coarse/fine fixed grids over raw latitude-longitude and "
        "train-fitted KMeans(32/96) pseudo-communities."
    )
    write_report(output_dir, metrics_md, best_line, seg_note)

    print(df.to_string(index=False))
    print(f"\nReport written to {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
