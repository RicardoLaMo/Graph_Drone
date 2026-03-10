from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from experiments.tabpfn_view_router.src.data import ViewData


PARTS = ("train", "val", "test")
SMOKE_LIMITS = {"train": 2048, "val": 512, "test": 512}


@dataclass(frozen=True)
class OpenMLRegressionSpec:
    key: str
    dataset_id: int
    task_id: int
    display_name: str
    target_name: str
    geo_columns: tuple[str, ...]
    domain_view_name: str = "DOMAIN"


OPENML_REGRESSION_SPECS: dict[str, OpenMLRegressionSpec] = {
    "diamonds": OpenMLRegressionSpec(
        key="diamonds",
        dataset_id=46923,
        task_id=363631,
        display_name="Diamonds",
        target_name="price",
        geo_columns=("carat", "depth", "table", "x", "y", "z"),
    ),
    "healthcare_insurance_expenses": OpenMLRegressionSpec(
        key="healthcare_insurance_expenses",
        dataset_id=46931,
        task_id=363675,
        display_name="Healthcare Insurance Expenses",
        target_name="charges",
        geo_columns=("age", "bmi", "children", "smoker"),
    ),
    "concrete_compressive_strength": OpenMLRegressionSpec(
        key="concrete_compressive_strength",
        dataset_id=46917,
        task_id=363625,
        display_name="Concrete Compressive Strength",
        target_name="ConcreteCompressiveStrength",
        geo_columns=("Cement", "BlastFurnaceSlag", "FlyAsh", "Water", "Superplasticizer"),
    ),
    "airfoil_self_noise": OpenMLRegressionSpec(
        key="airfoil_self_noise",
        dataset_id=46904,
        task_id=363612,
        display_name="Airfoil Self Noise",
        target_name="scaled-sound-pressure",
        geo_columns=("frequency", "attack-angle", "chord-length"),
    ),
    "wine_quality": OpenMLRegressionSpec(
        key="wine_quality",
        dataset_id=46964,
        task_id=363708,
        display_name="Wine Quality",
        target_name="median_wine_quality",
        geo_columns=(
            "fixed_acidity",
            "volatile_acidity",
            "citric_acid",
            "residual_sugar",
            "chlorides",
            "density",
            "pH",
            "alcohol",
        ),
    ),
    "used_fiat_500": OpenMLRegressionSpec(
        key="used_fiat_500",
        dataset_id=46907,
        task_id=363615,
        display_name="Used Fiat 500",
        target_name="price",
        geo_columns=("lat", "lon", "km", "age_in_days"),
    ),
    "miami_housing": OpenMLRegressionSpec(
        key="miami_housing",
        dataset_id=46942,
        task_id=363686,
        display_name="Miami Housing",
        target_name="SALE_PRC",
        geo_columns=(
            "LATITUDE",
            "LONGITUDE",
            "RAIL_DIST",
            "OCEAN_DIST",
            "WATER_DIST",
            "CNTR_DIST",
            "SUBCNTR_DI",
            "HWY_DIST",
        ),
    ),
    "houses": OpenMLRegressionSpec(
        key="houses",
        dataset_id=46934,
        task_id=363678,
        display_name="Houses",
        target_name="LnMedianHouseValue",
        geo_columns=("Latitude", "Longitude"),
    ),
}

OPENML_REGRESSION_ALIASES = {
    "Another-Dataset-on-used-Fiat-500": "used_fiat_500",
}


@dataclass(frozen=True)
class PreparedOpenMLSplit:
    dataset_key: str
    dataset_name: str
    dataset_id: int
    task_id: int
    target_name: str
    repeat: int
    fold: int
    split_seed: int
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    X_num_train: np.ndarray | None
    X_num_val: np.ndarray | None
    X_num_test: np.ndarray | None
    X_cat_train: np.ndarray | None
    X_cat_val: np.ndarray | None
    X_cat_test: np.ndarray | None
    feature_names: tuple[str, ...]
    num_feature_names: tuple[str, ...]
    cat_feature_names: tuple[str, ...]
    view_columns: dict[str, tuple[int, ...]]


def normalize_dataset_key(dataset_key: str) -> str:
    return OPENML_REGRESSION_ALIASES.get(dataset_key, dataset_key)


def available_dataset_keys() -> list[str]:
    return sorted(set(OPENML_REGRESSION_SPECS) | set(OPENML_REGRESSION_ALIASES))


def get_openml_regression_spec(dataset_key: str) -> OpenMLRegressionSpec:
    dataset_key = normalize_dataset_key(dataset_key)
    try:
        return OPENML_REGRESSION_SPECS[dataset_key]
    except KeyError as exc:
        known = ", ".join(available_dataset_keys())
        raise ValueError(f"Unknown dataset_key={dataset_key!r}; expected one of: {known}") from exc


def build_openml_regression_split(
    dataset_key: str,
    *,
    repeat: int,
    fold: int,
    split_seed: int = 42,
    val_fraction: float = 0.2,
    smoke: bool = False,
) -> PreparedOpenMLSplit:
    spec = get_openml_regression_spec(dataset_key)
    task = openml.tasks.get_task(spec.task_id)
    dataset = task.get_dataset()
    X_df, y_series, categorical_indicator, feature_names = dataset.get_data(
        dataset_format="dataframe",
        target=task.target_name,
    )

    num_feature_names = tuple(name for name, is_cat in zip(feature_names, categorical_indicator) if not is_cat)
    cat_feature_names = tuple(name for name, is_cat in zip(feature_names, categorical_indicator) if is_cat)

    train_full_idx, test_idx = task.get_train_test_split_indices(fold=fold, repeat=repeat, sample=0)
    train_idx, val_idx = train_test_split(
        train_full_idx,
        test_size=val_fraction,
        random_state=split_seed,
        shuffle=True,
    )
    train_idx = np.sort(np.asarray(train_idx, dtype=np.int64))
    val_idx = np.sort(np.asarray(val_idx, dtype=np.int64))
    test_idx = np.sort(np.asarray(test_idx, dtype=np.int64))

    if smoke:
        train_idx = _sample_indices(train_idx, SMOKE_LIMITS["train"], split_seed)
        val_idx = _sample_indices(val_idx, SMOKE_LIMITS["val"], split_seed + 1)
        test_idx = _sample_indices(test_idx, SMOKE_LIMITS["test"], split_seed + 2)

    X_num_parts = _extract_numeric_parts(X_df, num_feature_names, train_idx, val_idx, test_idx)
    X_cat_parts = _extract_categorical_parts(X_df, cat_feature_names, train_idx, val_idx, test_idx)
    X_encoded_parts, encoded_feature_names = encode_feature_parts(
        X_num_parts=X_num_parts,
        X_cat_parts=X_cat_parts,
        num_feature_names=num_feature_names,
        cat_feature_names=cat_feature_names,
    )
    view_columns = build_view_columns(encoded_feature_names, spec.geo_columns, spec.domain_view_name)

    y_all = y_series.to_numpy(dtype=np.float32)
    return PreparedOpenMLSplit(
        dataset_key=spec.key,
        dataset_name=spec.display_name,
        dataset_id=spec.dataset_id,
        task_id=spec.task_id,
        target_name=spec.target_name,
        repeat=repeat,
        fold=fold,
        split_seed=split_seed,
        X_train=X_encoded_parts["train"],
        X_val=X_encoded_parts["val"],
        X_test=X_encoded_parts["test"],
        y_train=y_all[train_idx].astype(np.float32),
        y_val=y_all[val_idx].astype(np.float32),
        y_test=y_all[test_idx].astype(np.float32),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        X_num_train=None if X_num_parts is None else X_num_parts["train"],
        X_num_val=None if X_num_parts is None else X_num_parts["val"],
        X_num_test=None if X_num_parts is None else X_num_parts["test"],
        X_cat_train=None if X_cat_parts is None else X_cat_parts["train"],
        X_cat_val=None if X_cat_parts is None else X_cat_parts["val"],
        X_cat_test=None if X_cat_parts is None else X_cat_parts["test"],
        feature_names=encoded_feature_names,
        num_feature_names=num_feature_names,
        cat_feature_names=cat_feature_names,
        view_columns=view_columns,
    )


def limit_train_rows(
    split: PreparedOpenMLSplit,
    *,
    max_train_samples: int,
    seed: int,
) -> PreparedOpenMLSplit:
    if max_train_samples <= 0 or len(split.y_train) <= max_train_samples:
        return split
    rng = np.random.RandomState(seed)
    keep = np.sort(rng.choice(len(split.y_train), size=max_train_samples, replace=False)).astype(np.int64)
    return replace(
        split,
        X_train=split.X_train[keep],
        y_train=split.y_train[keep],
        train_idx=split.train_idx[keep],
        X_num_train=None if split.X_num_train is None else split.X_num_train[keep],
        X_cat_train=None if split.X_cat_train is None else split.X_cat_train[keep],
    )


def encode_feature_parts(
    *,
    X_num_parts: dict[str, np.ndarray] | None,
    X_cat_parts: dict[str, np.ndarray] | None,
    num_feature_names: tuple[str, ...],
    cat_feature_names: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    encoded_parts: list[dict[str, np.ndarray]] = []
    ordered_feature_names: list[str] = []

    if X_num_parts is not None:
        encoded_num = _impute_numeric_parts(X_num_parts)
        encoded_parts.append(encoded_num)
        ordered_feature_names.extend(num_feature_names)

    if X_cat_parts is not None:
        encoded_cat = _encode_categorical_parts(X_cat_parts)
        encoded_parts.append(encoded_cat)
        ordered_feature_names.extend(cat_feature_names)

    if not encoded_parts:
        raise ValueError("At least one numeric or categorical feature block is required")

    merged: dict[str, np.ndarray] = {}
    for part in PARTS:
        blocks = [chunk[part] for chunk in encoded_parts]
        merged[part] = np.concatenate(blocks, axis=1).astype(np.float32)

    return merged, tuple(ordered_feature_names)


def build_view_columns(
    feature_names: tuple[str, ...],
    geo_columns: tuple[str, ...],
    domain_view_name: str,
) -> dict[str, tuple[int, ...]]:
    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    missing = tuple(name for name in geo_columns if name not in feature_to_idx)
    if missing:
        raise ValueError(f"Missing geo columns {missing!r} for feature_names={feature_names!r}")
    geo_idx = tuple(feature_to_idx[name] for name in geo_columns if name in feature_to_idx)
    if not geo_idx:
        raise ValueError(f"No geo columns from {geo_columns!r} found in feature_names={feature_names!r}")
    domain_idx = tuple(idx for idx, name in enumerate(feature_names) if name not in set(geo_columns))
    if not domain_idx:
        raise ValueError("Domain view would be empty; adjust geo column selection")
    return {"FULL": tuple(range(len(feature_names))), "GEO": geo_idx, domain_view_name: domain_idx}


def build_graphdrone_view_data(split: PreparedOpenMLSplit, *, lowrank_dim: int = 4) -> ViewData:
    from sklearn.decomposition import PCA

    n_components = min(lowrank_dim, split.X_train.shape[1], len(split.X_train))
    if n_components < 1:
        raise ValueError("Cannot build LOWRANK view with zero components")

    pca = PCA(n_components=n_components, random_state=split.split_seed)
    pca.fit(split.X_train)

    domain_view_name = next(name for name in split.view_columns if name not in {"FULL", "GEO"})
    view_names = ["FULL", "GEO", domain_view_name, "LOWRANK"]
    geo_idx = split.view_columns["GEO"]
    domain_idx = split.view_columns[domain_view_name]

    return ViewData(
        train={
            "FULL": split.X_train,
            "GEO": split.X_train[:, geo_idx],
            domain_view_name: split.X_train[:, domain_idx],
            "LOWRANK": pca.transform(split.X_train).astype(np.float32),
        },
        val={
            "FULL": split.X_val,
            "GEO": split.X_val[:, geo_idx],
            domain_view_name: split.X_val[:, domain_idx],
            "LOWRANK": pca.transform(split.X_val).astype(np.float32),
        },
        test={
            "FULL": split.X_test,
            "GEO": split.X_test[:, geo_idx],
            domain_view_name: split.X_test[:, domain_idx],
            "LOWRANK": pca.transform(split.X_test).astype(np.float32),
        },
        view_names=view_names,
    )


def write_foundation_dataset(output_dir: Path, split: PreparedOpenMLSplit) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if split.X_num_train is not None:
        np.save(output_dir / "X_num_train.npy", split.X_num_train)
        np.save(output_dir / "X_num_val.npy", split.X_num_val)
        np.save(output_dir / "X_num_test.npy", split.X_num_test)
    if split.X_cat_train is not None:
        np.save(output_dir / "X_cat_train.npy", split.X_cat_train)
        np.save(output_dir / "X_cat_val.npy", split.X_cat_val)
        np.save(output_dir / "X_cat_test.npy", split.X_cat_test)
    np.save(output_dir / "Y_train.npy", split.y_train)
    np.save(output_dir / "Y_val.npy", split.y_val)
    np.save(output_dir / "Y_test.npy", split.y_test)
    info = {
        "task_type": "regression",
        "score": "rmse",
        "name": split.dataset_name,
        "id": f"openml-{split.dataset_id}-task-{split.task_id}-r{split.repeat}f{split.fold}",
        "split_policy": f"openml_task_{split.task_id}_repeat{split.repeat}_fold{split.fold}_inner_val_seed{split.split_seed}",
    }
    (output_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    (output_dir / "READY").write_text("")
    return output_dir


def dataset_run_tag(dataset_key: str, *, repeat: int, fold: int, smoke: bool = False) -> str:
    dataset_key = normalize_dataset_key(dataset_key)
    tag = f"{dataset_key}__r{repeat}f{fold}"
    if smoke:
        tag = f"{tag}__smoke"
    return tag


def split_summary(split: PreparedOpenMLSplit) -> dict[str, object]:
    return {
        "dataset_key": split.dataset_key,
        "dataset_name": split.dataset_name,
        "dataset_id": split.dataset_id,
        "task_id": split.task_id,
        "target_name": split.target_name,
        "repeat": split.repeat,
        "fold": split.fold,
        "split_seed": split.split_seed,
        "train_rows": int(len(split.y_train)),
        "val_rows": int(len(split.y_val)),
        "test_rows": int(len(split.y_test)),
        "num_features": len(split.num_feature_names),
        "cat_features": len(split.cat_feature_names),
        "feature_names": list(split.feature_names),
    }


def _sample_indices(indices: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(indices) <= limit:
        return indices.astype(np.int64)
    rng = np.random.RandomState(seed)
    sampled = rng.choice(indices, size=limit, replace=False)
    sampled.sort()
    return sampled.astype(np.int64)


def _extract_numeric_parts(
    X_df,
    feature_names: tuple[str, ...],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, np.ndarray] | None:
    if not feature_names:
        return None
    values = X_df.loc[:, list(feature_names)].to_numpy(dtype=np.float32)
    return {
        "train": values[train_idx].astype(np.float32),
        "val": values[val_idx].astype(np.float32),
        "test": values[test_idx].astype(np.float32),
    }


def _extract_categorical_parts(
    X_df,
    feature_names: tuple[str, ...],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, np.ndarray] | None:
    if not feature_names:
        return None
    cat_df = X_df.loc[:, list(feature_names)].copy()
    cat_df = cat_df.astype(object).where(~cat_df.isna(), "__missing__")
    values = cat_df.astype(str).to_numpy(dtype=np.str_)
    return {
        "train": values[train_idx],
        "val": values[val_idx],
        "test": values[test_idx],
    }


def _impute_numeric_parts(X_num_parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    medians = np.nanmedian(X_num_parts["train"], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)

    def transform(block: np.ndarray) -> np.ndarray:
        out = block.astype(np.float32, copy=True)
        mask = np.isnan(out)
        if mask.any():
            out[mask] = np.take(medians, np.where(mask)[1])
        return out.astype(np.float32)

    return {part: transform(values) for part, values in X_num_parts.items()}


def _encode_categorical_parts(X_cat_parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.int64,
    )
    encoder.fit(X_cat_parts["train"])
    encoded: dict[str, np.ndarray] = {
        part: encoder.transform(values).astype(np.int64)
        for part, values in X_cat_parts.items()
    }
    max_values = encoded["train"].max(axis=0) if encoded["train"].shape[1] else np.array([], dtype=np.int64)
    for part in ("val", "test"):
        values = encoded[part]
        for col_idx in range(values.shape[1]):
            unknown_mask = values[:, col_idx] < 0
            if unknown_mask.any():
                values[unknown_mask, col_idx] = int(max_values[col_idx]) + 1
    return {part: values.astype(np.float32) for part, values in encoded.items()}
