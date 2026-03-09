from __future__ import annotations

import numpy as np

from experiments.california_geo_segmentation.src.segmentation import (
    GridSegmenter,
    KMeansSegmenter,
    build_segmentation_result,
)


def test_grid_segmenter_assigns_ids():
    geo = np.array(
        [
            [34.0, -118.0],
            [34.2, -118.1],
            [38.5, -121.4],
            [38.7, -121.6],
        ],
        dtype=np.float32,
    )
    seg = GridSegmenter(0.5, 0.5).fit(geo[:2])
    ids = seg.predict(geo)
    assert ids.shape == (4,)
    assert ids.dtype == np.int64


def test_kmeans_segmentation_result_has_expected_features():
    rng = np.random.default_rng(42)
    geo = np.concatenate(
        [
            rng.normal(loc=(34.0, -118.0), scale=0.05, size=(20, 2)),
            rng.normal(loc=(38.0, -122.0), scale=0.05, size=(20, 2)),
        ],
        axis=0,
    ).astype(np.float32)
    y = rng.normal(size=40).astype(np.float32)
    train_idx = np.arange(30, dtype=np.int64)
    result = build_segmentation_result("kmeans4", KMeansSegmenter(4), geo, y, train_idx)
    assert result.features.shape[0] == 40
    assert result.features.shape[1] == 5
    assert "kmeans4_centroid_dist" in result.feature_names


def test_segment_priors_are_finite():
    geo = np.array(
        [[34.0, -118.0], [34.0, -118.0], [34.5, -118.2], [38.1, -122.2]],
        dtype=np.float32,
    )
    y = np.array([1.0, 1.5, 2.0, 3.0], dtype=np.float32)
    train_idx = np.array([0, 1, 2], dtype=np.int64)
    result = build_segmentation_result("grid", GridSegmenter(0.25, 0.25), geo, y, train_idx)
    assert np.isfinite(result.features).all()


def test_test_targets_do_not_affect_train_only_segment_priors():
    geo = np.array(
        [[34.0, -118.0], [34.2, -118.1], [38.5, -121.4], [38.7, -121.6]],
        dtype=np.float32,
    )
    y1 = np.array([1.0, 2.0, 10.0, 20.0], dtype=np.float32)
    y2 = np.array([1.0, 2.0, -99.0, 999.0], dtype=np.float32)
    train_idx = np.array([0, 1], dtype=np.int64)
    r1 = build_segmentation_result("grid", GridSegmenter(0.5, 0.5), geo, y1, train_idx)
    r2 = build_segmentation_result("grid", GridSegmenter(0.5, 0.5), geo, y2, train_idx)
    np.testing.assert_allclose(r1.features, r2.features)
