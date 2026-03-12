from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


PARTS = ("train", "val", "test")
SMOKE_LIMITS = {"train": 2048, "val": 512, "test": 512}


@dataclass(frozen=True)
class OpenMLClassificationSpec:
    key: str
    dataset_id: int
    task_id: int
    display_name: str
    target_name: str


OPENML_CLASSIFICATION_SPECS: dict[str, OpenMLClassificationSpec] = {
    "diabetes": OpenMLClassificationSpec(
        key="diabetes",
        dataset_id=46921,
        task_id=363629,
        display_name="Diabetes",
        target_name="TestedPositiveForDiabetes",
    ),
    "anneal": OpenMLClassificationSpec(
        key="anneal",
        dataset_id=46906,
        task_id=363614,
        display_name="Anneal",
        target_name="classes",
    ),
    "maternal_health_risk": OpenMLClassificationSpec(
        key="maternal_health_risk",
        dataset_id=46941,
        task_id=363685,
        display_name="Maternal Health Risk",
        target_name="RiskLevel",
    ),
    "website_phishing": OpenMLClassificationSpec(
        key="website_phishing",
        dataset_id=46963,
        task_id=363707,
        display_name="Website Phishing",
        target_name="WebsiteType",
    ),
    "bioresponse": OpenMLClassificationSpec(
        key="bioresponse",
        dataset_id=46912,
        task_id=363620,
        display_name="Bioresponse",
        target_name="MoleculeElicitsResponse",
    ),
    "students_dropout_and_academic_success": OpenMLClassificationSpec(
        key="students_dropout_and_academic_success",
        dataset_id=46960,
        task_id=363704,
        display_name="Students Dropout And Academic Success",
        target_name="AcademicOutcome",
    ),
    "bank_customer_churn": OpenMLClassificationSpec(
        key="bank_customer_churn",
        dataset_id=46911,
        task_id=363619,
        display_name="Bank Customer Churn",
        target_name="churn",
    ),
    "bank_marketing": OpenMLClassificationSpec(
        key="bank_marketing",
        dataset_id=46910,
        task_id=363618,
        display_name="Bank Marketing",
        target_name="SubscribeTermDeposit",
    ),
    "apsfailure": OpenMLClassificationSpec(
        key="apsfailure",
        dataset_id=46908,
        task_id=363616,
        display_name="APSFailure",
        target_name="AirPressureSystemFailure",
    ),
}


OPENML_CLASSIFICATION_ALIASES = {
    "APSFailure": "apsfailure",
    "Bank_Customer_Churn": "bank_customer_churn",
    "bank-marketing": "bank_marketing",
    "Bioresponse": "bioresponse",
}


@dataclass(frozen=True)
class PreparedOpenMLClassificationSplit:
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
    class_labels: tuple[int, ...]
    class_names: tuple[str, ...]


def normalize_dataset_key(dataset_key: str) -> str:
    return OPENML_CLASSIFICATION_ALIASES.get(dataset_key, dataset_key)


def available_dataset_keys() -> list[str]:
    return sorted(set(OPENML_CLASSIFICATION_SPECS) | set(OPENML_CLASSIFICATION_ALIASES))


def get_openml_classification_spec(dataset_key: str) -> OpenMLClassificationSpec:
    dataset_key = normalize_dataset_key(dataset_key)
    try:
        return OPENML_CLASSIFICATION_SPECS[dataset_key]
    except KeyError as exc:
        known = ", ".join(available_dataset_keys())
        raise ValueError(f"Unknown dataset_key={dataset_key!r}; expected one of: {known}") from exc


def build_openml_classification_split(
    dataset_key: str,
    *,
    repeat: int,
    fold: int,
    split_seed: int = 42,
    val_fraction: float = 0.2,
    smoke: bool = False,
) -> PreparedOpenMLClassificationSplit:
    spec = get_openml_classification_spec(dataset_key)
    task = openml.tasks.get_task(spec.task_id)
    X_df, y_series, num_feature_names, cat_feature_names, feature_names = _load_openml_dataframe_with_retry(
        task=task,
        spec=spec,
    )

    label_encoder = LabelEncoder()
    y_raw = y_series.astype(str).to_numpy()
    y_all = label_encoder.fit_transform(y_raw).astype(np.int64)
    class_names = tuple(str(v) for v in label_encoder.classes_.tolist())
    class_labels = tuple(int(v) for v in range(len(class_names)))

    train_full_idx, test_idx = task.get_train_test_split_indices(fold=fold, repeat=repeat, sample=0)
    train_idx, val_idx = train_test_split(
        train_full_idx,
        test_size=val_fraction,
        random_state=split_seed,
        shuffle=True,
        stratify=y_all[train_full_idx],
    )
    train_idx = np.sort(np.asarray(train_idx, dtype=np.int64))
    val_idx = np.sort(np.asarray(val_idx, dtype=np.int64))
    test_idx = np.sort(np.asarray(test_idx, dtype=np.int64))

    if smoke:
        train_idx = _stratified_sample_indices(train_idx, y_all[train_idx], SMOKE_LIMITS["train"], split_seed)
        val_idx = _stratified_sample_indices(val_idx, y_all[val_idx], SMOKE_LIMITS["val"], split_seed + 1)
        test_idx = _stratified_sample_indices(test_idx, y_all[test_idx], SMOKE_LIMITS["test"], split_seed + 2)

    X_num_parts = _extract_numeric_parts(X_df, num_feature_names, train_idx, val_idx, test_idx)
    X_cat_parts = _extract_categorical_parts(X_df, cat_feature_names, train_idx, val_idx, test_idx)
    X_encoded_parts, encoded_feature_names = encode_feature_parts(
        X_num_parts=X_num_parts,
        X_cat_parts=X_cat_parts,
        num_feature_names=num_feature_names,
        cat_feature_names=cat_feature_names,
    )

    return PreparedOpenMLClassificationSplit(
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
        y_train=y_all[train_idx].astype(np.int64),
        y_val=y_all[val_idx].astype(np.int64),
        y_test=y_all[test_idx].astype(np.int64),
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
        class_labels=class_labels,
        class_names=class_names,
    )


def limit_train_rows(
    split: PreparedOpenMLClassificationSplit,
    *,
    max_train_samples: int,
    seed: int,
) -> PreparedOpenMLClassificationSplit:
    if max_train_samples <= 0 or len(split.y_train) <= max_train_samples:
        return split
    keep = _stratified_sample_indices(
        np.arange(len(split.y_train), dtype=np.int64),
        split.y_train,
        max_train_samples,
        seed,
    )
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
        merged[part] = np.concatenate([chunk[part] for chunk in encoded_parts], axis=1).astype(np.float32)
    return merged, tuple(ordered_feature_names)


def write_foundation_dataset(output_dir: Path, split: PreparedOpenMLClassificationSplit) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_foundation_arrays(
        output_dir=output_dir,
        X_num_parts=None
        if split.X_num_train is None
        else {
            "train": split.X_num_train,
            "val": split.X_num_val,
            "test": split.X_num_test,
        },
        X_cat_parts=None
        if split.X_cat_train is None
        else {
            "train": split.X_cat_train,
            "val": split.X_cat_val,
            "test": split.X_cat_test,
        },
        y_parts={
            "train": split.y_train.astype(np.int64),
            "val": split.y_val.astype(np.int64),
            "test": split.y_test.astype(np.int64),
        },
    )
    _write_foundation_info(output_dir=output_dir, split=split)
    return output_dir


def write_tabr_foundation_dataset(output_dir: Path, split: PreparedOpenMLClassificationSplit) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    X_num_parts = (
        None
        if split.X_num_train is None
        else {
            "train": split.X_num_train,
            "val": split.X_num_val,
            "test": split.X_num_test,
        }
    )
    X_cat_parts = (
        None
        if split.X_cat_train is None
        else {
            "train": split.X_cat_train,
            "val": split.X_cat_val,
            "test": split.X_cat_test,
        }
    )
    y_parts = {
        "train": split.y_train.astype(np.int64),
        "val": split.y_val.astype(np.int64),
        "test": split.y_test.astype(np.int64),
    }
    encoded_cat_parts, augmented_num_parts, augmented_y_parts = _encode_tabr_categorical_parts(
        X_cat_parts=X_cat_parts,
        X_num_parts=X_num_parts,
        y_parts=y_parts,
    )
    _write_foundation_arrays(
        output_dir=output_dir,
        X_num_parts=augmented_num_parts,
        X_cat_parts=encoded_cat_parts,
        y_parts=augmented_y_parts,
    )
    _write_foundation_info(output_dir=output_dir, split=split)
    return output_dir


def dataset_run_tag(dataset_key: str, *, repeat: int, fold: int, smoke: bool = False) -> str:
    dataset_key = normalize_dataset_key(dataset_key)
    tag = f"{dataset_key}__r{repeat}f{fold}"
    if smoke:
        tag = f"{tag}__smoke"
    return tag


def split_summary(split: PreparedOpenMLClassificationSplit) -> dict[str, object]:
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
        "n_classes": len(split.class_labels),
        "class_labels": list(split.class_labels),
        "class_names": list(split.class_names),
    }


def _stratified_sample_indices(
    indices: np.ndarray,
    y: np.ndarray,
    limit: int,
    seed: int,
) -> np.ndarray:
    if len(indices) <= limit:
        return indices.astype(np.int64)
    keep, _ = train_test_split(
        indices,
        train_size=limit,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )
    keep = np.sort(np.asarray(keep, dtype=np.int64))
    return keep


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

    return {part: transform(block) for part, block in X_num_parts.items()}


def _encode_categorical_parts(X_cat_parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
        dtype=np.float32,
    )
    encoder.fit(X_cat_parts["train"])
    return {
        part: encoder.transform(block).astype(np.float32)
        for part, block in X_cat_parts.items()
    }


def _write_foundation_arrays(
    *,
    output_dir: Path,
    X_num_parts: dict[str, np.ndarray] | None,
    X_cat_parts: dict[str, np.ndarray] | None,
    y_parts: dict[str, np.ndarray],
) -> None:
    if X_num_parts is not None:
        np.save(output_dir / "X_num_train.npy", X_num_parts["train"])
        np.save(output_dir / "X_num_val.npy", X_num_parts["val"])
        np.save(output_dir / "X_num_test.npy", X_num_parts["test"])
    if X_cat_parts is not None:
        np.save(output_dir / "X_cat_train.npy", X_cat_parts["train"])
        np.save(output_dir / "X_cat_val.npy", X_cat_parts["val"])
        np.save(output_dir / "X_cat_test.npy", X_cat_parts["test"])
    np.save(output_dir / "Y_train.npy", y_parts["train"].astype(np.int64))
    np.save(output_dir / "Y_val.npy", y_parts["val"].astype(np.int64))
    np.save(output_dir / "Y_test.npy", y_parts["test"].astype(np.int64))


def _write_foundation_info(*, output_dir: Path, split: PreparedOpenMLClassificationSplit) -> None:
    task_type = "binclass" if len(split.class_labels) == 2 else "multiclass"
    score = "roc-auc" if len(split.class_labels) == 2 else "accuracy"
    info = {
        "task_type": task_type,
        "score": score,
        "name": split.dataset_name,
        "id": f"openml-{split.dataset_id}-task-{split.task_id}-r{split.repeat}f{split.fold}",
        "split_policy": f"openml_task_{split.task_id}_repeat{split.repeat}_fold{split.fold}_inner_val_seed{split.split_seed}",
    }
    (output_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    (output_dir / "READY").write_text("")


def _load_openml_dataframe_with_retry(
    *,
    task,
    spec: OpenMLClassificationSpec,
) -> tuple[pd.DataFrame, pd.Series, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    dataset = task.get_dataset()
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            X_df, y_series, categorical_indicator, feature_names = dataset.get_data(
                dataset_format="dataframe",
                target=task.target_name,
            )
            repaired_df, num_feature_names, cat_feature_names = _repair_feature_partitions(
                X_df,
                feature_names=tuple(feature_names),
                categorical_indicator=tuple(bool(v) for v in categorical_indicator),
            )
            return repaired_df, y_series, num_feature_names, cat_feature_names, tuple(feature_names)
        except Exception as exc:
            last_exc = exc
            if attempt == 0 and _looks_like_openml_cache_issue(exc):
                _clear_openml_dataset_cache(spec.dataset_id)
                task = openml.tasks.get_task(spec.task_id)
                dataset = task.get_dataset()
                continue
            raise
    assert last_exc is not None
    raise last_exc


def _looks_like_openml_cache_issue(exc: Exception) -> bool:
    message = str(exc)
    cache_markers = (
        "Parquet magic bytes",
        "Bad @DATA instance format",
        "Could not download file",
        "dataset_",
        ".pq",
        ".arff",
        ".part.minio",
        "No such file or directory",
    )
    return any(marker in message for marker in cache_markers)


def _clear_openml_dataset_cache(dataset_id: int) -> None:
    cache_root = Path(openml.config.get_cache_directory())
    dataset_cache_dir = cache_root / "org" / "openml" / "www" / "datasets" / str(dataset_id)
    if dataset_cache_dir.exists():
        shutil.rmtree(dataset_cache_dir)


def _repair_feature_partitions(
    X_df: pd.DataFrame,
    *,
    feature_names: tuple[str, ...],
    categorical_indicator: tuple[bool, ...],
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    repaired = X_df.copy()
    num_feature_names: list[str] = []
    cat_feature_names: list[str] = []

    for name, is_cat in zip(feature_names, categorical_indicator):
        column = repaired[name]
        if is_cat:
            cat_feature_names.append(name)
            continue
        if pd.api.types.is_numeric_dtype(column):
            num_feature_names.append(name)
            continue
        coerced = pd.to_numeric(column, errors="coerce")
        non_missing = ~column.isna()
        failed = coerced.isna() & non_missing
        if failed.any():
            cat_feature_names.append(name)
            continue
        repaired[name] = coerced
        num_feature_names.append(name)

    return repaired, tuple(num_feature_names), tuple(cat_feature_names)


def _encode_tabr_categorical_parts(
    *,
    X_cat_parts: dict[str, np.ndarray] | None,
    X_num_parts: dict[str, np.ndarray] | None,
    y_parts: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray] | None, dict[str, np.ndarray] | None, dict[str, np.ndarray]]:
    if X_cat_parts is None:
        return None, X_num_parts, y_parts

    train = X_cat_parts["train"].astype(str, copy=False)
    val = X_cat_parts["val"].astype(str, copy=False)
    test = X_cat_parts["test"].astype(str, copy=False)
    n_train, n_cat = train.shape
    encoded_train = np.zeros((n_train + 1, n_cat), dtype=np.int64)
    encoded_val = np.zeros((len(val), n_cat), dtype=np.int64)
    encoded_test = np.zeros((len(test), n_cat), dtype=np.int64)

    for column_idx in range(n_cat):
        vocab: dict[str, int] = {}
        next_code = 1
        for value in train[:, column_idx]:
            if value not in vocab:
                vocab[value] = next_code
                next_code += 1
        encoded_train[1:, column_idx] = np.asarray([vocab[value] for value in train[:, column_idx]], dtype=np.int64)
        encoded_val[:, column_idx] = np.asarray([vocab.get(value, 0) for value in val[:, column_idx]], dtype=np.int64)
        encoded_test[:, column_idx] = np.asarray([vocab.get(value, 0) for value in test[:, column_idx]], dtype=np.int64)

    augmented_y_parts = {
        "train": np.concatenate([y_parts["train"][:1], y_parts["train"]], axis=0),
        "val": y_parts["val"],
        "test": y_parts["test"],
    }
    if X_num_parts is None:
        augmented_num_parts = None
    else:
        anchor_row = X_num_parts["train"][:1].astype(np.float32, copy=True)
        augmented_num_parts = {
            "train": np.concatenate([anchor_row, X_num_parts["train"]], axis=0),
            "val": X_num_parts["val"],
            "test": X_num_parts["test"],
        }

    encoded_cat_parts = {
        "train": encoded_train,
        "val": encoded_val,
        "test": encoded_test,
    }
    return encoded_cat_parts, augmented_num_parts, augmented_y_parts
