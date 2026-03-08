from experiments.tab_foundation_compare.src.aligned_california import (
    SEED,
    build_aligned_california_split,
    standardize_regression_targets,
)


def test_aligned_split_sizes_sum_to_dataset_size():
    split = build_aligned_california_split(SEED)
    total = len(split.train_idx) + len(split.val_idx) + len(split.test_idx)
    assert total == 20640
    assert len(split.train_idx) == 14448
    assert len(split.val_idx) == 3096
    assert len(split.test_idx) == 3096


def test_target_standardization_round_trip_shape():
    split = build_aligned_california_split(SEED)
    y_train, y_val, y_test, stats = standardize_regression_targets(
        split.y_train, split.y_val, split.y_test
    )
    assert y_train.shape == split.y_train.shape
    assert y_val.shape == split.y_val.shape
    assert y_test.shape == split.y_test.shape
    assert abs(float(y_train.mean())) < 1e-4
    assert 0.99 < float(y_train.std()) < 1.01
    assert stats["std"] > 0.0
