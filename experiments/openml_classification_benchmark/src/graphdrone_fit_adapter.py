from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.graphdrone_fit.expert_factory import (
    ExpertBuildSpec,
    GeometryFeatureAdapter,
    IdentitySelectorAdapter,
    PcaProjectionAdapter,
)
from src.graphdrone_fit.token_builder import QualityEncoding
from src.graphdrone_fit.view_descriptor import ViewDescriptor, normalize_descriptor_set

from .openml_tasks import PreparedOpenMLClassificationSplit


@dataclass(frozen=True)
class GraphDroneClassificationExpertPlan:
    dataset_key: str
    descriptors: tuple[ViewDescriptor, ...]
    specs: tuple[ExpertBuildSpec, ...]
    full_expert_id: str
    expert_role_map: dict[str, str]


QUALITY_FEATURE_NAMES = (
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


def build_classification_expert_plan(
    split: PreparedOpenMLClassificationSplit,
    *,
    seed: int,
    n_estimators: int,
    n_preprocessing_jobs: int,
    device: str,
) -> GraphDroneClassificationExpertPlan:
    descriptors = _build_descriptors(split)
    specs: list[ExpertBuildSpec] = []
    expert_role_map: dict[str, str] = {}

    for descriptor in descriptors:
        expert_role_map[descriptor.expert_id] = descriptor.view_name
        if descriptor.expert_id == "ANCHOR":
            adapter = IdentitySelectorAdapter(indices=descriptor.input_indices)
        elif descriptor.projection_kind == "identity_subselect":
            adapter = IdentitySelectorAdapter(indices=descriptor.input_indices)
        elif descriptor.view_name == "LOWRANK":
            adapter = PcaProjectionAdapter(
                n_components=descriptor.input_dim,
                random_state=split.split_seed,
            )
        elif descriptor.view_name == "GEOMETRY_LID":
            adapter = GeometryFeatureAdapter(
                indices=tuple(range(split.X_train.shape[1])),
                feature_keys=("lid", "mean_knn_distance"),
                include_base_features=True,
                k_neighbors=min(24, max(4, len(split.y_train) - 1)),
            )
        elif descriptor.view_name == "GEOMETRY_LOF":
            adapter = GeometryFeatureAdapter(
                indices=tuple(range(split.X_train.shape[1])),
                feature_keys=("lof", "mean_knn_distance"),
                include_base_features=True,
                k_neighbors=min(24, max(4, len(split.y_train) - 1)),
            )
        else:
            raise ValueError(f"Unsupported descriptor view_name={descriptor.view_name!r}")

        specs.append(
            ExpertBuildSpec(
                descriptor=descriptor,
                model_kind="tabpfn_classifier",
                input_adapter=adapter,
                model_params={
                    "n_estimators": n_estimators,
                    "random_state": seed,
                    "device": device,
                    "ignore_pretraining_limits": len(split.y_train) > 1000,
                    "n_preprocessing_jobs": n_preprocessing_jobs,
                },
            )
        )

    return GraphDroneClassificationExpertPlan(
        dataset_key=split.dataset_key,
        descriptors=descriptors,
        specs=tuple(specs),
        full_expert_id="ANCHOR",
        expert_role_map=expert_role_map,
    )


def build_classification_quality_encodings(
    split: PreparedOpenMLClassificationSplit,
    plan: GraphDroneClassificationExpertPlan,
    *,
    k_neighbors: int = 24,
) -> dict[str, QualityEncoding]:
    fitted_views: dict[str, dict[str, np.ndarray]] = {}
    for spec in plan.specs:
        adapter = spec.input_adapter.fit(split.X_train)
        fitted_views[spec.descriptor.expert_id] = {
            "train": adapter.transform(split.X_train).astype(np.float32),
            "val": adapter.transform(split.X_val).astype(np.float32),
            "test": adapter.transform(split.X_test).astype(np.float32),
        }

    encodings: dict[str, QualityEncoding] = {}
    for part in ("train", "val", "test"):
        idx_map, distance_map = _neighbor_maps(
            fitted_views=fitted_views,
            descriptors=plan.descriptors,
            split=split,
            part=part,
            k_neighbors=k_neighbors,
        )
        encodings[part] = _build_quality_encoding_for_part(
            idx_map=idx_map,
            distance_map=distance_map,
            descriptors=plan.descriptors,
            y_train=split.y_train,
            class_count=len(split.class_labels),
            fitted_views=fitted_views,
            part=part,
        )
    return encodings


def _build_descriptors(split: PreparedOpenMLClassificationSplit) -> tuple[ViewDescriptor, ...]:
    full_indices = tuple(range(split.X_train.shape[1]))
    descriptors: list[ViewDescriptor] = [
        ViewDescriptor(
            expert_id="ANCHOR",
            family="FULL",
            view_name="ANCHOR",
            projection_kind="identity_subselect",
            input_dim=len(full_indices),
            input_indices=full_indices,
            feature_names=split.feature_names,
            is_anchor=True,
            source_name=split.dataset_key,
            tags=("anchor", "classification", "full_features"),
        )
    ]

    if split.num_feature_names and len(split.num_feature_names) < len(split.feature_names):
        numeric_indices = tuple(range(len(split.num_feature_names)))
        descriptors.append(
            ViewDescriptor(
                expert_id="NUMERIC_1",
                family="numeric_subset",
                view_name="NUMERIC_ONLY",
                projection_kind="identity_subselect",
                input_dim=len(numeric_indices),
                input_indices=numeric_indices,
                feature_names=split.num_feature_names,
                source_name=split.dataset_key,
                tags=("classification", "numeric_subset"),
            )
        )

    if split.cat_feature_names and len(split.cat_feature_names) < len(split.feature_names):
        offset = len(split.num_feature_names)
        categorical_indices = tuple(range(offset, offset + len(split.cat_feature_names)))
        descriptors.append(
            ViewDescriptor(
                expert_id="CATEGORICAL_1",
                family="categorical_subset",
                view_name="CATEGORICAL_ONLY",
                projection_kind="identity_subselect",
                input_dim=len(categorical_indices),
                input_indices=categorical_indices,
                feature_names=split.cat_feature_names,
                source_name=split.dataset_key,
                tags=("classification", "categorical_subset"),
            )
        )

    lowrank_dim = min(8, split.X_train.shape[1], len(split.y_train))
    if lowrank_dim >= 2:
        descriptors.append(
            ViewDescriptor(
                expert_id="SUBSPACE_1",
                family="structural_subspace",
                view_name="LOWRANK",
                projection_kind="external_transform",
                input_dim=lowrank_dim,
                feature_names=tuple(f"lowrank_{idx}" for idx in range(lowrank_dim)),
                source_name=split.dataset_key,
                tags=("classification", "lowrank"),
            )
        )

    full_feature_names = split.feature_names
    geometry_defs = (
        ("GEOMETRY_1", "GEOMETRY_LID", ("lid", "mean_knn_distance")),
        ("GEOMETRY_2", "GEOMETRY_LOF", ("lof", "mean_knn_distance")),
    )
    for expert_id, view_name, feature_keys in geometry_defs:
        feature_names = tuple(full_feature_names) + tuple(f"geometry_{key}" for key in feature_keys)
        descriptors.append(
            ViewDescriptor(
                expert_id=expert_id,
                family="geometry_signal",
                view_name=view_name,
                projection_kind="derived_features",
                input_dim=len(feature_names),
                feature_names=feature_names,
                source_name=split.dataset_key,
                tags=("classification", "geometry", *(f"geometry_{key}" for key in feature_keys)),
            )
        )

    return normalize_descriptor_set(descriptors, required_anchor_id="ANCHOR")


def _neighbor_maps(
    *,
    fitted_views: dict[str, dict[str, np.ndarray]],
    descriptors: tuple[ViewDescriptor, ...],
    split: PreparedOpenMLClassificationSplit,
    part: str,
    k_neighbors: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    idx_map: dict[str, np.ndarray] = {}
    distance_map: dict[str, np.ndarray] = {}
    for descriptor in descriptors:
        train_view = fitted_views[descriptor.expert_id]["train"]
        query_view = fitted_views[descriptor.expert_id][part]
        drop_self = part == "train"
        n_neighbors = min(max(k_neighbors + (1 if drop_self else 0), 2), len(train_view))
        knn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
        knn.fit(train_view)
        distances, indices = knn.kneighbors(query_view, return_distance=True)
        if drop_self and distances.shape[1] > 1:
            distances = distances[:, 1:]
            indices = indices[:, 1:]
        idx_map[descriptor.expert_id] = indices.astype(np.int64)
        distance_map[descriptor.expert_id] = distances.astype(np.float32)
    return idx_map, distance_map


def _build_quality_encoding_for_part(
    *,
    idx_map: dict[str, np.ndarray],
    distance_map: dict[str, np.ndarray],
    descriptors: tuple[ViewDescriptor, ...],
    y_train: np.ndarray,
    class_count: int,
    fitted_views: dict[str, dict[str, np.ndarray]],
    part: str,
) -> QualityEncoding:
    n_rows = next(iter(idx_map.values())).shape[0]
    n_experts = len(descriptors)
    class_ids = np.arange(class_count, dtype=np.int64)

    entropy_cols: list[np.ndarray] = []
    confidence_cols: list[np.ndarray] = []
    margin_cols: list[np.ndarray] = []
    mean_distance_cols: list[np.ndarray] = []
    neighbor_sets: list[np.ndarray] = []
    geometry_lid_cols: list[np.ndarray] = []
    geometry_lof_cols: list[np.ndarray] = []
    geometry_distance_cols: list[np.ndarray] = []

    for descriptor in descriptors:
        expert_id = descriptor.expert_id
        indices = idx_map[expert_id]
        distances = np.maximum(distance_map[expert_id], 1e-8)
        neighbor_labels = y_train[indices]
        probs = (neighbor_labels[..., None] == class_ids).mean(axis=1).astype(np.float32)
        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-8)
        top_k = np.sort(probs, axis=1)[:, ::-1]
        entropy_cols.append((-(probs * np.log(probs)).sum(axis=1)).astype(np.float32))
        confidence_cols.append(top_k[:, 0].astype(np.float32))
        if class_count == 1:
            margin_cols.append(np.zeros(n_rows, dtype=np.float32))
        else:
            margin_cols.append((top_k[:, 0] - top_k[:, 1]).astype(np.float32))
        mean_distance = distances.mean(axis=1).astype(np.float32)
        mean_distance_cols.append(mean_distance)
        neighbor_sets.append(indices)

        transformed = fitted_views[expert_id][part]
        if descriptor.view_name == "GEOMETRY_LID":
            geometry_lid_cols.append(transformed[:, -2].astype(np.float32))
            geometry_lof_cols.append(np.zeros(n_rows, dtype=np.float32))
            geometry_distance_cols.append(transformed[:, -1].astype(np.float32))
        elif descriptor.view_name == "GEOMETRY_LOF":
            geometry_lid_cols.append(np.zeros(n_rows, dtype=np.float32))
            geometry_lof_cols.append(transformed[:, -2].astype(np.float32))
            geometry_distance_cols.append(transformed[:, -1].astype(np.float32))
        else:
            geometry_lid_cols.append(np.zeros(n_rows, dtype=np.float32))
            geometry_lof_cols.append(np.zeros(n_rows, dtype=np.float32))
            geometry_distance_cols.append(np.zeros(n_rows, dtype=np.float32))

    j_matrix = np.zeros((n_rows, n_experts, n_experts), dtype=np.float32)
    for i in range(n_experts):
        left = np.sort(neighbor_sets[i], axis=1)
        left_k = left.shape[1]
        for j in range(i + 1, n_experts):
            right = np.sort(neighbor_sets[j], axis=1)
            right_k = right.shape[1]
            inter = np.zeros(n_rows, dtype=np.float32)
            for row_idx in range(n_rows):
                inter[row_idx] = float(np.intersect1d(left[row_idx], right[row_idx], assume_unique=False).size)
            union = np.maximum(float(left_k + right_k) - inter, 1.0)
            score = inter / union
            j_matrix[:, i, j] = score
            j_matrix[:, j, i] = score

    if n_experts == 1:
        pair_mean = np.zeros((n_rows, 1), dtype=np.float32)
        pair_max = np.zeros((n_rows, 1), dtype=np.float32)
    else:
        pair_mean = (j_matrix.sum(axis=2) / float(n_experts - 1)).astype(np.float32)
        pair_max = j_matrix.max(axis=2).astype(np.float32)
    mean_j_global = pair_mean.mean(axis=1, keepdims=True).astype(np.float32)
    mean_j_broadcast = np.repeat(mean_j_global, n_experts, axis=1)

    tensor = np.stack(
        [
            np.stack(entropy_cols, axis=1),
            np.stack(confidence_cols, axis=1),
            np.stack(margin_cols, axis=1),
            pair_mean,
            pair_max,
            mean_j_broadcast,
            np.stack(geometry_lid_cols, axis=1),
            np.stack(geometry_lof_cols, axis=1),
            np.stack(geometry_distance_cols, axis=1),
        ],
        axis=-1,
    ).astype(np.float32)
    return QualityEncoding(
        tensor=tensor,
        feature_names=QUALITY_FEATURE_NAMES,
    )
