from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from experiments.openml_regression_benchmark.src.graphdrone_fit_adapter import (
    build_benchmark_descriptors,
    build_benchmark_expert_plan,
    build_benchmark_quality_encodings,
)
from experiments.tabpfn_view_router.src.data import QualityFeatures, ViewData


def test_build_benchmark_descriptors_handles_variable_view_kinds() -> None:
    split = SimpleNamespace(
        dataset_key="california_housing_openml",
        dataset_name="California Housing",
        dataset_id=44024,
        task_id=362499,
        target_name="median_house_value",
        repeat=0,
        fold=0,
        split_seed=42,
        X_train=np.ones((4, 5), dtype=np.float32),
        X_val=np.ones((2, 5), dtype=np.float32),
        X_test=np.ones((2, 5), dtype=np.float32),
        y_train=np.ones(4, dtype=np.float32),
        y_val=np.ones(2, dtype=np.float32),
        y_test=np.ones(2, dtype=np.float32),
        train_idx=np.arange(4, dtype=np.int64),
        val_idx=np.arange(2, dtype=np.int64),
        test_idx=np.arange(2, dtype=np.int64),
        X_num_train=np.ones((4, 5), dtype=np.float32),
        X_num_val=np.ones((2, 5), dtype=np.float32),
        X_num_test=np.ones((2, 5), dtype=np.float32),
        X_cat_train=None,
        X_cat_val=None,
        X_cat_test=None,
        feature_names=("MedInc", "HouseAge", "Latitude", "Longitude", "Population"),
        num_feature_names=("MedInc", "HouseAge", "Latitude", "Longitude", "Population"),
        cat_feature_names=(),
        view_columns={"FULL": (0, 1, 2, 3, 4), "GEO": (2, 3), "SOCIO": (0, 1, 4)},
    )
    views = ViewData(
        train={
            "FULL": np.ones((4, 5), dtype=np.float32),
            "GEO": np.ones((4, 2), dtype=np.float32),
            "SOCIO": np.ones((4, 3), dtype=np.float32),
            "LOWRANK": np.ones((4, 2), dtype=np.float32),
        },
        val={
            "FULL": np.ones((2, 5), dtype=np.float32),
            "GEO": np.ones((2, 2), dtype=np.float32),
            "SOCIO": np.ones((2, 3), dtype=np.float32),
            "LOWRANK": np.ones((2, 2), dtype=np.float32),
        },
        test={
            "FULL": np.ones((2, 5), dtype=np.float32),
            "GEO": np.ones((2, 2), dtype=np.float32),
            "SOCIO": np.ones((2, 3), dtype=np.float32),
            "LOWRANK": np.ones((2, 2), dtype=np.float32),
        },
        view_names=["FULL", "GEO", "SOCIO", "LOWRANK"],
    )

    descriptor_set = build_benchmark_descriptors(split, views)
    ids = [descriptor.expert_id for descriptor in descriptor_set.descriptors]
    assert ids == ["FULL", "GEO", "SOCIO", "LOWRANK"]
    full = descriptor_set.descriptors[0]
    geo = descriptor_set.descriptors[1]
    lowrank = descriptor_set.descriptors[-1]
    assert full.is_anchor is True
    assert geo.family == "domain_semantic"
    assert lowrank.projection_kind == "external_transform"


def test_build_benchmark_descriptors_accepts_explicit_family_overrides() -> None:
    split = SimpleNamespace(
        dataset_key="houses",
        dataset_name="Houses",
        dataset_id=46934,
        task_id=363678,
        target_name="LnMedianHouseValue",
        repeat=0,
        fold=0,
        split_seed=42,
        X_train=np.ones((3, 4), dtype=np.float32),
        X_val=np.ones((1, 4), dtype=np.float32),
        X_test=np.ones((1, 4), dtype=np.float32),
        y_train=np.ones(3, dtype=np.float32),
        y_val=np.ones(1, dtype=np.float32),
        y_test=np.ones(1, dtype=np.float32),
        train_idx=np.arange(3, dtype=np.int64),
        val_idx=np.arange(1, dtype=np.int64),
        test_idx=np.arange(1, dtype=np.int64),
        X_num_train=np.ones((3, 4), dtype=np.float32),
        X_num_val=np.ones((1, 4), dtype=np.float32),
        X_num_test=np.ones((1, 4), dtype=np.float32),
        X_cat_train=None,
        X_cat_val=None,
        X_cat_test=None,
        feature_names=("MedianIncome", "Latitude", "Longitude", "Population"),
        num_feature_names=("MedianIncome", "Latitude", "Longitude", "Population"),
        cat_feature_names=(),
        view_columns={"FULL": (0, 1, 2, 3), "GEO": (1, 2), "DOMAIN": (0, 3)},
    )
    views = ViewData(
        train={
            "FULL": np.ones((3, 4), dtype=np.float32),
            "GEO": np.ones((3, 2), dtype=np.float32),
            "DOMAIN": np.ones((3, 2), dtype=np.float32),
            "LOWRANK": np.ones((3, 2), dtype=np.float32),
        },
        val={
            "FULL": np.ones((1, 4), dtype=np.float32),
            "GEO": np.ones((1, 2), dtype=np.float32),
            "DOMAIN": np.ones((1, 2), dtype=np.float32),
            "LOWRANK": np.ones((1, 2), dtype=np.float32),
        },
        test={
            "FULL": np.ones((1, 4), dtype=np.float32),
            "GEO": np.ones((1, 2), dtype=np.float32),
            "DOMAIN": np.ones((1, 2), dtype=np.float32),
            "LOWRANK": np.ones((1, 2), dtype=np.float32),
        },
        view_names=["FULL", "GEO", "DOMAIN", "LOWRANK"],
    )

    descriptor_set = build_benchmark_descriptors(
        split,
        views,
        family_overrides={
            "GEO": "domain_semantic",
            "DOMAIN": "domain_semantic",
            "LOWRANK": "learned_regime",
        },
    )
    assert descriptor_set.descriptors[1].family == "domain_semantic"
    assert descriptor_set.descriptors[2].family == "domain_semantic"
    assert descriptor_set.descriptors[3].family == "learned_regime"


def test_build_benchmark_expert_plan_creates_tabpfn_specs() -> None:
    split = SimpleNamespace(
        dataset_key="houses",
        dataset_name="Houses",
        dataset_id=46934,
        task_id=363678,
        target_name="LnMedianHouseValue",
        repeat=0,
        fold=0,
        split_seed=42,
        X_train=np.ones((3, 4), dtype=np.float32),
        X_val=np.ones((1, 4), dtype=np.float32),
        X_test=np.ones((1, 4), dtype=np.float32),
        y_train=np.ones(3, dtype=np.float32),
        y_val=np.ones(1, dtype=np.float32),
        y_test=np.ones(1, dtype=np.float32),
        train_idx=np.arange(3, dtype=np.int64),
        val_idx=np.arange(1, dtype=np.int64),
        test_idx=np.arange(1, dtype=np.int64),
        X_num_train=np.ones((3, 4), dtype=np.float32),
        X_num_val=np.ones((1, 4), dtype=np.float32),
        X_num_test=np.ones((1, 4), dtype=np.float32),
        X_cat_train=None,
        X_cat_val=None,
        X_cat_test=None,
        feature_names=("MedianIncome", "Latitude", "Longitude", "Population"),
        num_feature_names=("MedianIncome", "Latitude", "Longitude", "Population"),
        cat_feature_names=(),
        view_columns={"FULL": (0, 1, 2, 3), "GEO": (1, 2), "DOMAIN": (0, 3)},
    )
    views = ViewData(
        train={
            "FULL": np.ones((3, 4), dtype=np.float32),
            "GEO": np.ones((3, 2), dtype=np.float32),
            "DOMAIN": np.ones((3, 2), dtype=np.float32),
            "LOWRANK": np.ones((3, 2), dtype=np.float32),
        },
        val={
            "FULL": np.ones((1, 4), dtype=np.float32),
            "GEO": np.ones((1, 2), dtype=np.float32),
            "DOMAIN": np.ones((1, 2), dtype=np.float32),
            "LOWRANK": np.ones((1, 2), dtype=np.float32),
        },
        test={
            "FULL": np.ones((1, 4), dtype=np.float32),
            "GEO": np.ones((1, 2), dtype=np.float32),
            "DOMAIN": np.ones((1, 2), dtype=np.float32),
            "LOWRANK": np.ones((1, 2), dtype=np.float32),
        },
        view_names=["FULL", "GEO", "DOMAIN", "LOWRANK"],
    )

    plan = build_benchmark_expert_plan(
        split,
        views,
        seed=17,
        n_estimators=2,
        n_preprocessing_jobs=3,
        view_devices={"FULL": "cpu", "GEO": "cpu", "DOMAIN": "cpu", "LOWRANK": "cpu"},
        family_overrides={"GEO": "domain_semantic"},
    )
    assert len(plan.specs) == 4
    assert plan.specs[0].model_kind == "tabpfn_regressor"
    assert plan.specs[-1].descriptor.family == "structural_subspace"


def test_build_benchmark_quality_encodings_maps_legacy_flat_priors() -> None:
    views = ViewData(
        train={"FULL": np.ones((2, 4), dtype=np.float32), "GEO": np.ones((2, 2), dtype=np.float32)},
        val={"FULL": np.ones((2, 4), dtype=np.float32), "GEO": np.ones((2, 2), dtype=np.float32)},
        test={"FULL": np.ones((2, 4), dtype=np.float32), "GEO": np.ones((2, 2), dtype=np.float32)},
        view_names=["FULL", "GEO"],
    )
    quality = QualityFeatures(
        train=np.array([[0.2, 0.5, 0.8, 0.7], [0.1, 0.4, 0.7, 0.6]], dtype=np.float32),
        val=np.array([[0.3, 0.6, 0.9, 0.8], [0.2, 0.5, 0.8, 0.7]], dtype=np.float32),
        test=np.array([[0.4, 0.7, 1.0, 0.9], [0.3, 0.6, 0.9, 0.8]], dtype=np.float32),
        sigma2_train=np.zeros((2, 2), dtype=np.float32),
        sigma2_val=np.zeros((2, 2), dtype=np.float32),
        sigma2_test=np.zeros((2, 2), dtype=np.float32),
        mean_j_train=np.zeros(2, dtype=np.float32),
        mean_j_val=np.zeros(2, dtype=np.float32),
        mean_j_test=np.zeros(2, dtype=np.float32),
    )

    encodings = build_benchmark_quality_encodings(views, quality)
    assert set(encodings) == {"train", "val", "test"}
    assert encodings["train"].tensor.shape == (2, 2, 5)
    assert encodings["train"].feature_names[0] == "quality_sigma2_self"
