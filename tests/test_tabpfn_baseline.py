import importlib.util

import pytest

from experiments.tab_foundation_compare.src.aligned_california import build_aligned_california_split
from experiments.tab_foundation_compare.src.tabpfn_baseline import (
    evaluate_tabpfn_regression,
    select_train_subset,
)


def test_select_train_subset_respects_cap_and_seed():
    split = build_aligned_california_split()
    idx_a = select_train_subset(len(split.X_train), max_train_samples=1000, seed=42)
    idx_b = select_train_subset(len(split.X_train), max_train_samples=1000, seed=42)
    assert len(idx_a) == 1000
    assert len(set(idx_a.tolist())) == 1000
    assert idx_a.tolist() == idx_b.tolist()


def test_select_train_subset_returns_full_when_cap_is_none():
    split = build_aligned_california_split()
    idx = select_train_subset(len(split.X_train), max_train_samples=None, seed=42)
    assert len(idx) == len(split.X_train)
    assert idx[0] == 0
    assert idx[-1] == len(split.X_train) - 1


def test_evaluate_tabpfn_regression_smoke_returns_metrics():
    if importlib.util.find_spec("tabpfn") is None:
        pytest.skip("tabpfn not installed in active environment")
    split = build_aligned_california_split()
    metrics = evaluate_tabpfn_regression(
        split=split,
        seed=42,
        max_train_samples=256,
        n_estimators=2,
        max_eval_rows=64,
    )
    assert metrics["train_samples_used"] == 256
    assert metrics["n_estimators"] == 2
    assert metrics["test"]["rmse"] > 0.0
    assert metrics["val"]["rmse"] > 0.0
