from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from experiments.openml_classification_benchmark.src.graphdrone_fit_adapter import (
    build_classification_expert_plan,
    build_classification_quality_encodings,
)


def test_build_classification_expert_plan_creates_generic_roles() -> None:
    split = SimpleNamespace(
        dataset_key="bank_marketing",
        dataset_name="Bank Marketing",
        dataset_id=46910,
        task_id=363618,
        target_name="SubscribeTermDeposit",
        repeat=0,
        fold=0,
        split_seed=42,
        X_train=np.ones((8, 6), dtype=np.float32),
        X_val=np.ones((4, 6), dtype=np.float32),
        X_test=np.ones((4, 6), dtype=np.float32),
        y_train=np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64),
        y_val=np.array([0, 1, 0, 1], dtype=np.int64),
        y_test=np.array([0, 1, 0, 1], dtype=np.int64),
        X_num_train=np.ones((8, 4), dtype=np.float32),
        X_num_val=np.ones((4, 4), dtype=np.float32),
        X_num_test=np.ones((4, 4), dtype=np.float32),
        X_cat_train=np.ones((8, 2), dtype=np.float32),
        X_cat_val=np.ones((4, 2), dtype=np.float32),
        X_cat_test=np.ones((4, 2), dtype=np.float32),
        feature_names=("n0", "n1", "n2", "n3", "c0", "c1"),
        num_feature_names=("n0", "n1", "n2", "n3"),
        cat_feature_names=("c0", "c1"),
        class_labels=(0, 1),
        class_names=("no", "yes"),
    )
    plan = build_classification_expert_plan(
        split,
        seed=42,
        n_estimators=1,
        n_preprocessing_jobs=1,
        device="cpu",
    )
    expert_ids = [spec.descriptor.expert_id for spec in plan.specs]
    assert expert_ids[0] == "ANCHOR"
    assert "NUMERIC_1" in expert_ids
    assert "CATEGORICAL_1" in expert_ids
    assert "SUBSPACE_1" in expert_ids
    assert "GEOMETRY_1" in expert_ids
    assert "GEOMETRY_2" in expert_ids
    assert all(spec.model_kind == "tabpfn_classifier" for spec in plan.specs)


def test_build_classification_quality_encodings_has_expected_shape_and_fields() -> None:
    split = SimpleNamespace(
        dataset_key="bank_marketing",
        dataset_name="Bank Marketing",
        dataset_id=46910,
        task_id=363618,
        target_name="SubscribeTermDeposit",
        repeat=0,
        fold=0,
        split_seed=42,
        X_train=np.arange(48, dtype=np.float32).reshape(8, 6),
        X_val=np.arange(24, dtype=np.float32).reshape(4, 6),
        X_test=np.arange(24, dtype=np.float32).reshape(4, 6),
        y_train=np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64),
        y_val=np.array([0, 1, 0, 1], dtype=np.int64),
        y_test=np.array([0, 1, 0, 1], dtype=np.int64),
        X_num_train=np.arange(32, dtype=np.float32).reshape(8, 4),
        X_num_val=np.arange(16, dtype=np.float32).reshape(4, 4),
        X_num_test=np.arange(16, dtype=np.float32).reshape(4, 4),
        X_cat_train=np.arange(16, dtype=np.float32).reshape(8, 2),
        X_cat_val=np.arange(8, dtype=np.float32).reshape(4, 2),
        X_cat_test=np.arange(8, dtype=np.float32).reshape(4, 2),
        feature_names=("n0", "n1", "n2", "n3", "c0", "c1"),
        num_feature_names=("n0", "n1", "n2", "n3"),
        cat_feature_names=("c0", "c1"),
        class_labels=(0, 1),
        class_names=("no", "yes"),
    )
    plan = build_classification_expert_plan(
        split,
        seed=42,
        n_estimators=1,
        n_preprocessing_jobs=1,
        device="cpu",
    )
    encodings = build_classification_quality_encodings(split, plan, k_neighbors=3)
    assert set(encodings) == {"train", "val", "test"}
    for part_name, row_count in (("train", 8), ("val", 4), ("test", 4)):
        encoding = encodings[part_name]
        assert encoding.tensor.shape[:2] == (row_count, len(plan.specs))
        assert encoding.feature_names == (
            "quality_knn_entropy",
            "quality_knn_confidence",
            "quality_knn_margin",
            "quality_pair_overlap_mean",
            "quality_pair_overlap_max",
            "quality_mean_J_global",
            "quality_geometry_lid",
            "quality_geometry_lof",
            "quality_geometry_mean_knn_distance",
        )
