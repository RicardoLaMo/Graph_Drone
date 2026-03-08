from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.tabr_california_baseline.src.data_bridge import write_california_dataset


def test_write_california_dataset_creates_tabr_style_directory(tmp_path: Path):
    output_dir = tmp_path / "california_local"

    write_california_dataset(output_dir, seed=0)

    expected = {
        "info.json",
        "X_num_train.npy",
        "X_num_val.npy",
        "X_num_test.npy",
        "Y_train.npy",
        "Y_val.npy",
        "Y_test.npy",
    }
    assert expected.issubset({p.name for p in output_dir.iterdir()})

    info = json.loads((output_dir / "info.json").read_text())
    assert info["task_type"] == "regression"

    x_train = np.load(output_dir / "X_num_train.npy")
    y_train = np.load(output_dir / "Y_train.npy")
    x_val = np.load(output_dir / "X_num_val.npy")
    x_test = np.load(output_dir / "X_num_test.npy")
    assert x_train.ndim == 2
    assert x_train.shape[1] == 8
    assert y_train.ndim == 1
    assert x_train.shape[0] > x_val.shape[0] > 0
    assert x_train.shape[0] > x_test.shape[0] > 0

