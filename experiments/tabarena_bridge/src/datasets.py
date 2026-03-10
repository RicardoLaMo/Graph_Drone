from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import openml
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


@dataclass(frozen=True)
class Tier1Split:
    dataset_name: str
    task_id: int
    split_seed: int
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    n_features: int
    train_rows: int
    val_rows: int
    test_rows: int


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ],
        sparse_threshold=0.0,
    )


def build_openml_regression_split(task_id: int, split_seed: int) -> Tier1Split:
    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y(dataset_format="dataframe")
    y = np.asarray(y, dtype=np.float32)

    idx = np.arange(len(X))
    train_idx, tmp_idx = train_test_split(idx, test_size=0.30, random_state=split_seed)
    val_idx, test_idx = train_test_split(tmp_idx, test_size=0.50, random_state=split_seed)

    X_train_df = X.iloc[train_idx].copy()
    X_val_df = X.iloc[val_idx].copy()
    X_test_df = X.iloc[test_idx].copy()

    preprocessor = _build_preprocessor(X_train_df)
    X_train = preprocessor.fit_transform(X_train_df).astype(np.float32)
    X_val = preprocessor.transform(X_val_df).astype(np.float32)
    X_test = preprocessor.transform(X_test_df).astype(np.float32)

    return Tier1Split(
        dataset_name=str(task.get_dataset().name),
        task_id=int(task_id),
        split_seed=int(split_seed),
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y[train_idx],
        y_val=y[val_idx],
        y_test=y[test_idx],
        n_features=int(X_train.shape[1]),
        train_rows=int(len(train_idx)),
        val_rows=int(len(val_idx)),
        test_rows=int(len(test_idx)),
    )
