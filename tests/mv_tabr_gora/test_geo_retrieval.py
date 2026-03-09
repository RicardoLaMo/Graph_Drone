from __future__ import annotations

import numpy as np

from experiments.mv_tabr_gora.src.data import (
    _apply_same_segment_weight_bias,
    _build_random_poolmix_for_view,
    _build_segment_poolmix_for_view,
)


def test_same_segment_weight_bias_keeps_rows_normalized():
    idx = np.array([[1, 2, 3], [0, 2, 3]], dtype=np.int64)
    wt = np.array([[0.2, 0.3, 0.5], [0.4, 0.3, 0.3]], dtype=np.float32)
    seg = np.array([0, 0, 1, 1], dtype=np.int64)
    out = _apply_same_segment_weight_bias(
        {"GEO": (idx, wt)},
        seg,
        anchor_ids=np.array([0, 1], dtype=np.int64),
        target_views=("GEO",),
        bonus=2.0,
    )
    _, wt_new = out["GEO"]
    np.testing.assert_allclose(wt_new.sum(axis=1), np.ones(2, dtype=np.float32), atol=1e-6)


def test_segment_poolmix_returns_train_referenced_neighbors():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.2, 5.0],
        ],
        dtype=np.float32,
    )
    train_idx = np.arange(6, dtype=np.int64)
    seg = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    base_idx = np.array(
        [
            [1, 2],
            [0, 2],
            [1, 0],
            [4, 5],
            [3, 5],
            [4, 3],
        ],
        dtype=np.int64,
    )
    idx, wt = _build_segment_poolmix_for_view(X, base_idx, train_idx, seg, k=2, k_seg=1)
    assert idx.shape == (6, 2)
    assert wt.shape == (6, 2)
    assert np.isin(idx, train_idx).all()
    np.testing.assert_allclose(wt.sum(axis=1), np.ones(6, dtype=np.float32), atol=1e-6)


def test_random_poolmix_returns_train_referenced_neighbors():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.2, 5.0],
        ],
        dtype=np.float32,
    )
    train_idx = np.arange(6, dtype=np.int64)
    base_idx = np.array(
        [
            [1, 2],
            [0, 2],
            [1, 0],
            [4, 5],
            [3, 5],
            [4, 3],
        ],
        dtype=np.int64,
    )
    idx, wt = _build_random_poolmix_for_view(X, base_idx, train_idx, k=2, k_rand=1, seed=42)
    assert idx.shape == (6, 2)
    assert np.isin(idx, train_idx).all()
    np.testing.assert_allclose(wt.sum(axis=1), np.ones(6, dtype=np.float32), atol=1e-6)
