from __future__ import annotations

import numpy as np

from experiments.mv_tabr_gora.src.data import (
    _compute_segment_target_offsets,
    build_cohort_residual_bundle,
)


def test_segment_offsets_use_leave_one_out_for_train_rows():
    y = np.asarray([1.0, 3.0, 10.0, 12.0], dtype=np.float32)
    train_idx = np.asarray([0, 1, 2], dtype=np.int64)
    segment_ids = np.asarray([0, 0, 1, 1], dtype=np.int64)

    offsets = _compute_segment_target_offsets(y, train_idx, segment_ids, fallback_mean=5.0)

    assert np.isclose(offsets[0], 3.0)
    assert np.isclose(offsets[1], 1.0)
    assert np.isclose(offsets[2], 5.0)  # singleton train segment falls back
    assert np.isclose(offsets[3], 10.0)  # test row gets train-segment mean


def test_cohort_residual_bundle_appends_segment_mean_to_geo():
    bundle = build_cohort_residual_bundle(
        smoke=True,
        smoke_train=64,
        smoke_val=32,
        smoke_test=32,
        seed=0,
    )

    assert bundle.view_dims["GEO"] == 3
    assert bundle.view_feats["GEO"].shape[1] == 3
    assert bundle.target_offset.shape == bundle.y.shape
    assert bundle.geo_segment_ids is not None
    assert np.isfinite(bundle.y_norm).all()
