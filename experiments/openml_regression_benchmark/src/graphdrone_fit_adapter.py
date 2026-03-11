from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from experiments.tabpfn_view_router.src.data import QualityFeatures, ViewData
from src.graphdrone_fit.expert_factory import (
    ExpertBuildSpec,
    GeometryFeatureAdapter,
    IdentitySelectorAdapter,
    PcaProjectionAdapter,
)
from src.graphdrone_fit.token_builder import QualityEncoding, build_legacy_quality_encoding_from_flat
from src.graphdrone_fit.view_descriptor import ViewDescriptor, normalize_descriptor_set

if TYPE_CHECKING:
    from experiments.openml_regression_benchmark.src.openml_tasks import PreparedOpenMLSplit


@dataclass(frozen=True)
class GraphDroneBenchmarkDescriptorSet:
    dataset_key: str
    descriptors: tuple[ViewDescriptor, ...]


@dataclass(frozen=True)
class GraphDroneBenchmarkExpertPlan:
    dataset_key: str
    descriptors: tuple[ViewDescriptor, ...]
    specs: tuple[ExpertBuildSpec, ...]
    full_expert_id: str
    expert_view_map: dict[str, str]
    expert_quality_source: dict[str, str]


GEOMETRY_QUALITY_FEATURE_KEYS = ("lid", "lof", "mean_knn_distance")


def build_benchmark_descriptors(
    split: PreparedOpenMLSplit,
    views: ViewData,
    *,
    family_overrides: dict[str, str] | None = None,
) -> GraphDroneBenchmarkDescriptorSet:
    descriptors: list[ViewDescriptor] = []
    full_indices = tuple(range(split.X_train.shape[1]))
    family_overrides = dict(family_overrides or {})
    family_counts: dict[str, int] = {}

    for name in views.view_names:
        if name == "FULL":
            descriptors.append(
                ViewDescriptor(
                    expert_id="ANCHOR",
                    family="FULL",
                    view_name=name,
                    projection_kind="identity_subselect",
                    input_dim=len(full_indices),
                    input_indices=full_indices,
                    feature_names=split.feature_names,
                    is_anchor=True,
                    source_name=split.dataset_key,
                    tags=("anchor", "portfolio", f"registry_view_{_tagify(name)}"),
                )
            )
            continue

        family = family_overrides.get(name, _default_family_for_view(name))
        expert_id = _canonical_expert_id(family=family, family_counts=family_counts)
        if name == "LOWRANK":
            descriptors.append(
                ViewDescriptor(
                    expert_id=expert_id,
                    family=family,
                    view_name=name,
                    projection_kind="external_transform",
                    input_dim=views.train[name].shape[1],
                    feature_names=tuple(f"lowrank_{idx}" for idx in range(views.train[name].shape[1])),
                    source_name=split.dataset_key,
                    tags=("portfolio", "transformed", f"registry_view_{_tagify(name)}"),
                )
            )
            continue

        input_indices = split.view_columns[name]
        feature_names = tuple(split.feature_names[idx] for idx in input_indices)
        descriptors.append(
            ViewDescriptor(
                expert_id=expert_id,
                family=family,
                view_name=name,
                projection_kind="identity_subselect",
                input_dim=len(input_indices),
                input_indices=input_indices,
                feature_names=feature_names,
                source_name=split.dataset_key,
                tags=("portfolio", "registry", f"registry_view_{_tagify(name)}"),
            )
        )

    return GraphDroneBenchmarkDescriptorSet(
        dataset_key=split.dataset_key,
        descriptors=normalize_descriptor_set(descriptors, required_anchor_id="ANCHOR"),
    )


def _default_family_for_view(view_name: str) -> str:
    upper_name = view_name.upper()
    if upper_name == "FULL":
        return "FULL"
    if upper_name in {"LOWRANK", "PCA", "SVD"}:
        return "structural_subspace"
    if upper_name in {"GEO", "DOMAIN", "SOCIO"}:
        return "domain_semantic"
    if "LOCAL" in upper_name or "SUPPORT" in upper_name or "KNN" in upper_name:
        return "local_support"
    if "REGIME" in upper_name or "CLUSTER" in upper_name:
        return "learned_regime"
    return "bootstrap"


def _canonical_expert_id(*, family: str, family_counts: dict[str, int]) -> str:
    family_counts[family] = family_counts.get(family, 0) + 1
    index = family_counts[family]
    prefix = {
        "domain_semantic": "SEMANTIC",
        "structural_subspace": "SUBSPACE",
        "local_support": "SUPPORT",
        "learned_regime": "REGIME",
        "geometry_signal": "GEOMETRY",
        "bootstrap": "SPECIALIST",
        "FULL": "ANCHOR",
    }.get(family, "EXPERT")
    if prefix == "ANCHOR":
        return "ANCHOR"
    return f"{prefix}_{index}"


def _tagify(text: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in text).strip("_")


def build_benchmark_expert_plan(
    split: PreparedOpenMLSplit,
    views: ViewData,
    *,
    seed: int,
    n_estimators: int,
    n_preprocessing_jobs: int,
    view_devices: dict[str, str | list[str]],
    family_overrides: dict[str, str] | None = None,
) -> GraphDroneBenchmarkExpertPlan:
    descriptor_set = build_benchmark_descriptors(
        split,
        views,
        family_overrides=family_overrides,
    )
    descriptor_by_name = {descriptor.view_name: descriptor for descriptor in descriptor_set.descriptors}

    specs: list[ExpertBuildSpec] = []
    expert_view_map: dict[str, str] = {}
    expert_quality_source: dict[str, str] = {}
    for name in views.view_names:
        descriptor = descriptor_by_name[name]
        expert_view_map[descriptor.expert_id] = name
        expert_quality_source[descriptor.expert_id] = name
        if name == "LOWRANK":
            adapter = PcaProjectionAdapter(
                n_components=views.train[name].shape[1],
                random_state=split.split_seed,
            )
        else:
            adapter = IdentitySelectorAdapter(indices=descriptor.input_indices)
        specs.append(
            ExpertBuildSpec(
                descriptor=descriptor,
                model_kind="tabpfn_regressor",
                input_adapter=adapter,
                model_params={
                    "n_estimators": n_estimators,
                    "random_state": seed,
                    "device": view_devices[name],
                    "ignore_pretraining_limits": len(split.y_train) > 1000,
                    "n_preprocessing_jobs": n_preprocessing_jobs,
                },
            )
        )

    anchor_descriptor = descriptor_by_name["FULL"]
    geometry_specs = _build_geometry_specs(
        split=split,
        anchor_descriptor=anchor_descriptor,
        seed=seed,
        n_estimators=n_estimators,
        n_preprocessing_jobs=n_preprocessing_jobs,
        device=view_devices["FULL"],
    )
    for spec in geometry_specs:
        specs.append(spec)
        expert_view_map[spec.descriptor.expert_id] = spec.descriptor.view_name
        expert_quality_source[spec.descriptor.expert_id] = "FULL"

    return GraphDroneBenchmarkExpertPlan(
        dataset_key=split.dataset_key,
        descriptors=tuple(list(descriptor_set.descriptors) + [spec.descriptor for spec in geometry_specs]),
        specs=tuple(specs),
        full_expert_id="ANCHOR",
        expert_view_map=expert_view_map,
        expert_quality_source=expert_quality_source,
    )


def _build_geometry_specs(
    *,
    split: PreparedOpenMLSplit,
    anchor_descriptor: ViewDescriptor,
    seed: int,
    n_estimators: int,
    n_preprocessing_jobs: int,
    device: str | list[str],
) -> tuple[ExpertBuildSpec, ...]:
    base_names = anchor_descriptor.feature_names
    base_dim = len(base_names)
    geometry_defs = (
        ("GEOMETRY_LID", ("lid", "mean_knn_distance")),
        ("GEOMETRY_LOF", ("lof", "mean_knn_distance")),
    )
    specs: list[ExpertBuildSpec] = []
    for offset, (view_name, feature_keys) in enumerate(geometry_defs, start=1):
        adapter = GeometryFeatureAdapter(
            indices=anchor_descriptor.input_indices,
            feature_keys=feature_keys,
            include_base_features=True,
            k_neighbors=min(24, max(4, len(split.y_train) - 1)),
        )
        feature_names = adapter.output_feature_names(base_names)
        descriptor = ViewDescriptor(
            expert_id=f"GEOMETRY_{offset}",
            family="geometry_signal",
            view_name=view_name,
            projection_kind="derived_features",
            input_dim=adapter.output_dim(base_dim),
            feature_names=feature_names,
            source_name=split.dataset_key,
            tags=("portfolio", "geometry", *(f"geometry_{key}" for key in feature_keys)),
        )
        specs.append(
            ExpertBuildSpec(
                descriptor=descriptor,
                model_kind="tabpfn_regressor",
                input_adapter=adapter,
                model_params={
                    "n_estimators": n_estimators,
                    "random_state": seed + offset,
                    "device": device,
                    "ignore_pretraining_limits": len(split.y_train) > 1000,
                    "n_preprocessing_jobs": n_preprocessing_jobs,
                },
            )
        )
    return tuple(specs)


def build_benchmark_quality_encodings(
    views: ViewData,
    quality: QualityFeatures,
    expert_plan: GraphDroneBenchmarkExpertPlan,
) -> dict[str, QualityEncoding]:
    view_index = {name: idx for idx, name in enumerate(views.view_names)}
    expert_ids = tuple(descriptor.expert_id for descriptor in expert_plan.descriptors)
    geometry_specs = {
        spec.descriptor.expert_id: spec
        for spec in expert_plan.specs
        if isinstance(spec.input_adapter, GeometryFeatureAdapter)
    }

    def build_geometry_tensor(full_matrix: np.ndarray) -> np.ndarray:
        extra = np.zeros(
            (full_matrix.shape[0], len(expert_ids), len(GEOMETRY_QUALITY_FEATURE_KEYS)),
            dtype=np.float32,
        )
        full_feature_names = tuple(f"full_{idx}" for idx in range(full_matrix.shape[1]))
        for expert_index, expert_id in enumerate(expert_ids):
            spec = geometry_specs.get(expert_id)
            if spec is None:
                continue
            original = spec.input_adapter
            adapter = GeometryFeatureAdapter(
                indices=original.indices,
                feature_keys=original.feature_keys,
                include_base_features=original.include_base_features,
                k_neighbors=original.k_neighbors,
            ).fit(views.train["FULL"])
            transformed = adapter.transform(full_matrix)
            output_names = adapter.output_feature_names(full_feature_names)
            field_to_column = {name.removeprefix("geometry_"): idx for idx, name in enumerate(output_names)}
            for feature_pos, feature_key in enumerate(GEOMETRY_QUALITY_FEATURE_KEYS):
                column_index = field_to_column.get(feature_key)
                if column_index is not None:
                    extra[:, expert_index, feature_pos] = transformed[:, column_index]
        return extra

    def expand(flat_quality: np.ndarray, full_matrix: np.ndarray) -> QualityEncoding:
        base = build_legacy_quality_encoding_from_flat(
            view_names=views.view_names,
            flat_quality=flat_quality,
        )
        expanded_base = np.stack(
            [
                base.tensor[:, view_index[expert_plan.expert_quality_source[expert_id]], :].detach().cpu().numpy()
                for expert_id in expert_ids
            ],
            axis=1,
        ).astype(np.float32)
        geometry_extra = build_geometry_tensor(full_matrix)
        expanded = np.concatenate([expanded_base, geometry_extra], axis=-1).astype(np.float32)
        return QualityEncoding(
            tensor=expanded,
            feature_names=base.feature_names
            + tuple(f"quality_geometry_{name}" for name in GEOMETRY_QUALITY_FEATURE_KEYS),
        )

    return {
        "train": expand(quality.train, views.train["FULL"]),
        "val": expand(quality.val, views.val["FULL"]),
        "test": expand(quality.test, views.test["FULL"]),
    }
