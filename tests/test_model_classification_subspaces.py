from __future__ import annotations

from graphdrone_fit.model import GraphDrone


def test_binary_classification_subspaces_are_multiple_and_proper() -> None:
    views = GraphDrone._binary_classification_subspaces(8)
    assert len(views) >= 3
    assert all(1 < len(view) < 8 for view in views)
    assert len(set(views)) == len(views)


def test_binary_classification_subspaces_empty_for_tiny_feature_space() -> None:
    assert GraphDrone._binary_classification_subspaces(2) == ()
