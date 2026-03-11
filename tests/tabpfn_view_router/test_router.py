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
    quality_feature_names,
    sigma2_mix,
    summarize_router_diagnostics,
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


def test_quality_feature_names_match_four_view_layout() -> None:
    names = quality_feature_names(["FULL", "GEO", "SOCIO", "LOWRANK"])
    assert names == [
        "sigma2_FULL",
        "sigma2_GEO",
        "sigma2_SOCIO",
        "sigma2_LOWRANK",
        "J_FULL_GEO",
        "J_FULL_SOCIO",
        "J_FULL_LOWRANK",
        "J_GEO_SOCIO",
        "J_GEO_LOWRANK",
        "J_SOCIO_LOWRANK",
        "mean_J",
    ]


def test_summarize_router_diagnostics_reports_alignment_and_entropy() -> None:
    y_true = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    pred_views = np.array(
        [
            [0.0, 0.3, 0.4, 0.5],
            [1.4, 1.0, 1.2, 1.1],
            [2.2, 2.3, 2.0, 2.5],
        ],
        dtype=np.float32,
    )
    weights = np.array(
        [
            [0.80, 0.10, 0.05, 0.05],
            [0.10, 0.70, 0.10, 0.10],
            [0.10, 0.10, 0.70, 0.10],
        ],
        dtype=np.float32,
    )
    quality = np.array(
        [
            [0.1, 0.3, 0.4, 0.5, 0.6, 0.2, 0.2, 0.1, 0.1, 0.1, 0.2],
            [0.4, 0.1, 0.3, 0.2, 0.6, 0.2, 0.2, 0.1, 0.1, 0.1, 0.5],
            [0.4, 0.3, 0.1, 0.2, 0.6, 0.2, 0.2, 0.1, 0.1, 0.1, 0.7],
        ],
        dtype=np.float32,
    )

    summary = summarize_router_diagnostics(
        y_true=y_true,
        pred_views=pred_views,
        weights=weights,
        quality_features=quality,
        view_names=["FULL", "GEO", "SOCIO", "LOWRANK"],
        anchor_view="FULL",
    )

    assert summary["n_rows"] == 3
    assert summary["anchor_view"] == "FULL"
    assert summary["top_weight_matches_oracle_best_fraction"] == 1.0
    assert summary["top_weight_fraction"]["FULL"] == 1.0 / 3.0
    assert summary["oracle_best_fraction"]["SOCIO"] == 1.0 / 3.0
    np.testing.assert_allclose(summary["mean_weight_when_oracle_best"]["GEO"], 0.7)
    assert summary["anchor_oracle_rmse_gap"] > 0.0
