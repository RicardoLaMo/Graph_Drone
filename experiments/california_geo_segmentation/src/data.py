from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from sklearn.datasets import fetch_california_housing

WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKTREE_ROOT))

from experiments.head_routing_v5.california.src.data_v5 import build_california_dataset
from experiments.head_routing_v5.california.src.train_v5 import maybe_truncate_splits


@dataclass(frozen=True)
class GeoDataBundle:
    X: np.ndarray
    y: np.ndarray
    raw_geo: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    feature_names: tuple[str, ...]
    target_stats: dict[str, float]


def build_geo_data_bundle(
    seed: int = 42,
    smoke: bool = False,
    smoke_train: int = 500,
    smoke_val: int = 200,
    smoke_test: int = 200,
) -> GeoDataBundle:
    ds = build_california_dataset(seed=seed)
    raw = fetch_california_housing()
    raw_geo = raw.data[:, [6, 7]].astype(np.float32)

    train_idx, val_idx, test_idx = maybe_truncate_splits(
        ds["train_idx"],
        ds["val_idx"],
        ds["test_idx"],
        smoke,
        smoke_train,
        smoke_val,
        smoke_test,
    )

    return GeoDataBundle(
        X=ds["X"],
        y=ds["y"],
        raw_geo=raw_geo,
        train_idx=np.asarray(train_idx, dtype=np.int64),
        val_idx=np.asarray(val_idx, dtype=np.int64),
        test_idx=np.asarray(test_idx, dtype=np.int64),
        feature_names=tuple(raw.feature_names),
        target_stats=ds["target_stats"],
    )
