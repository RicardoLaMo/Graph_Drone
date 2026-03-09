import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
V4_SRC = REPO_ROOT / "experiments" / "mq_gora_v4" / "shared" / "src"
if str(V4_SRC) not in sys.path:
    sys.path.insert(0, str(V4_SRC))

from eval_v4 import compute_routing_stats, write_routing_figures
from integrity_check import write_integrity_report


def test_compute_routing_stats_includes_beta_columns():
    pi = np.array(
        [
            [[0.8, 0.2], [0.4, 0.6]],
            [[0.7, 0.3], [0.2, 0.8]],
            [[0.6, 0.4], [0.1, 0.9]],
        ],
        dtype=np.float32,
    )
    beta = np.array(
        [
            [0.1, 0.8],
            [0.2, 0.7],
            [0.3, 0.6],
        ],
        dtype=np.float32,
    )
    tau = np.array([1.0, 1.5], dtype=np.float32)

    df = compute_routing_stats(pi, beta, tau, ["FULL", "PCA"], n_heads=2)

    assert "mean_beta" in df.columns
    assert "beta_std" in df.columns
    assert "routing_entropy" in df.columns
    assert "mean_pi_FULL" in df.columns
    assert "top1_freq_PCA" in df.columns


def test_write_routing_figures_creates_required_files(tmp_path):
    routing_df = pd.DataFrame(
        [
            {
                "head_idx": 0,
                "routing_entropy": 0.3,
                "dominant_view": "FULL",
                "mean_beta": 0.2,
                "beta_std": 0.1,
                "tau": 1.1,
                "mean_pi_FULL": 0.7,
                "mean_pi_PCA": 0.3,
                "top1_freq_FULL": 0.8,
                "top1_freq_PCA": 0.2,
            },
            {
                "head_idx": 1,
                "routing_entropy": 0.5,
                "dominant_view": "PCA",
                "mean_beta": 0.8,
                "beta_std": 0.1,
                "tau": 1.3,
                "mean_pi_FULL": 0.4,
                "mean_pi_PCA": 0.6,
                "top1_freq_FULL": 0.3,
                "top1_freq_PCA": 0.7,
            },
        ]
    )
    beta_by_regime = pd.DataFrame(
        [
            {"regime": "low", "head_idx": 0, "mean_beta": 0.2},
            {"regime": "high", "head_idx": 0, "mean_beta": 0.8},
            {"regime": "low", "head_idx": 1, "mean_beta": 0.4},
            {"regime": "high", "head_idx": 1, "mean_beta": 0.7},
        ]
    )
    view_similarity = pd.DataFrame(
        [[1.0, 0.3], [0.3, 1.0]],
        index=["FULL", "PCA"],
        columns=["FULL", "PCA"],
    )

    write_routing_figures(
        fig_dir=tmp_path,
        routing_df=routing_df,
        beta_by_regime_df=beta_by_regime,
        view_similarity=view_similarity,
        dataset_prefix="toy",
    )

    for name in [
        "pi_by_head.png",
        "top1_view_freq.png",
        "beta_distribution.png",
        "view_context_similarity.png",
        "tau_distribution.png",
    ]:
        assert (tmp_path / name).exists(), name


def test_write_integrity_report_mentions_self_alignment_and_references(tmp_path, monkeypatch):
    monkeypatch.setattr("integrity_check._REP_DIR", str(tmp_path))
    monkeypatch.setattr("integrity_check._ART_DIR", str(tmp_path))

    compat_rows = [
        {
            "model": "GoraTransformer",
            "accepts_view_mask": "Y",
            "accepts_z_anc": "Y",
            "accepts_lbl_nei": "Y",
            "accepts_agree_score": "Y",
            "status": "PASS",
            "note": "ok",
        }
    ]
    timing_rows = [
        {
            "fn": "compute_label_ctx_per_view",
            "shape": "(3, 4, 2)",
            "has_nan": "False",
            "time_s": "0.0001",
            "status": "PASS",
        }
    ]
    sanity_rows = [{"model": "MQGoraV4", "status": "PASS", "note": "ok"}]
    reference_rows = [
        {
            "model": "G2_GoRA_v1",
            "metric": "rmse",
            "current": 0.45,
            "reference": 0.45,
            "delta": 0.0,
            "status": "MATCH",
        }
    ]

    write_integrity_report(compat_rows, timing_rows, sanity_rows, reference_rows)

    report_text = (tmp_path / "system_integrity_report.md").read_text()
    assert "## Step 0 Self-Alignment" in report_text
    assert "## D. Reference Reproduction" in report_text
    assert "known bug fixes are numerically invariant" in report_text
