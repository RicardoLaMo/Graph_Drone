from __future__ import annotations

import numpy as np

from experiments.tabpfn_view_router.src.data import (
    build_aligned_california_split,
    build_quality_features,
    build_view_data,
)
from experiments.tabpfn_view_router.src.router import (
    fit_crossfit_router,
    fit_soft_router,
    sigma2_mix,
    uniform_mix,
)


def test_view_data_shapes_and_names() -> None:
    split = build_aligned_california_split(smoke=True)
    views = build_view_data(split)
    assert views.view_names == ["FULL", "GEO", "SOCIO", "LOWRANK"]
    assert views.train["FULL"].shape[1] == 8
    assert views.train["GEO"].shape[1] == 2
    assert views.train["SOCIO"].shape[1] == 5
    assert views.train["LOWRANK"].shape[1] == 4


def test_quality_feature_shape_matches_expected() -> None:
    split = build_aligned_california_split(smoke=True)
    views = build_view_data(split)
    quality = build_quality_features(split, views, k=8)
    assert quality.train.shape[1] == 11
    assert quality.val.shape[1] == 11
    assert quality.test.shape[1] == 11


def test_router_weights_are_valid_probabilities() -> None:
    rng = np.random.default_rng(0)
    x_val = rng.normal(size=(80, 11)).astype(np.float32)
    pred_val = rng.normal(size=(80, 4)).astype(np.float32)
    y_val = rng.normal(size=(80,)).astype(np.float32)
    x_test = rng.normal(size=(40, 11)).astype(np.float32)
    pred_test = rng.normal(size=(40, 4)).astype(np.float32)

    result = fit_soft_router(x_val, pred_val, y_val, x_test, pred_test, seed=0, max_epochs=20, patience=5)
    assert np.allclose(result.weights_val.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(result.weights_test.sum(axis=1), 1.0, atol=1e-5)


def test_crossfit_router_returns_valid_oof_and_test_weights() -> None:
    rng = np.random.default_rng(1)
    x_val = rng.normal(size=(80, 11)).astype(np.float32)
    pred_val = rng.normal(size=(80, 4)).astype(np.float32)
    y_val = rng.normal(size=(80,)).astype(np.float32)
    x_test = rng.normal(size=(40, 11)).astype(np.float32)
    pred_test = rng.normal(size=(40, 4)).astype(np.float32)

    result = fit_crossfit_router(
        x_val,
        pred_val,
        y_val,
        x_test,
        pred_test,
        n_splits=5,
        seed=0,
        max_epochs=20,
        patience=5,
    )

    assert result.weights_val_oof.shape == pred_val.shape
    assert result.weights_test.shape == pred_test.shape
    assert np.allclose(result.weights_val_oof.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(result.weights_test.sum(axis=1), 1.0, atol=1e-5)


def test_uniform_and_sigma2_mix_shapes() -> None:
    preds = np.ones((10, 4), dtype=np.float32)
    sigma2 = np.ones((10, 4), dtype=np.float32)
    p_u, w_u = uniform_mix(preds)
    p_s, w_s = sigma2_mix(preds, sigma2)
    assert p_u.shape == (10,)
    assert p_s.shape == (10,)
    assert w_u.shape == (10, 4)
    assert w_s.shape == (10, 4)
