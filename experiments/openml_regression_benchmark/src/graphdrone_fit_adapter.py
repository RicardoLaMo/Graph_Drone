from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from experiments.tabpfn_view_router.src.data import ViewData
from src.graphdrone_fit.view_descriptor import ViewDescriptor, normalize_descriptor_set

if TYPE_CHECKING:
    from experiments.openml_regression_benchmark.src.openml_tasks import PreparedOpenMLSplit


@dataclass(frozen=True)
class GraphDroneBenchmarkDescriptorSet:
    dataset_key: str
    descriptors: tuple[ViewDescriptor, ...]


def build_benchmark_descriptors(
    split: PreparedOpenMLSplit,
    views: ViewData,
    *,
    family_overrides: dict[str, str] | None = None,
) -> GraphDroneBenchmarkDescriptorSet:
    descriptors: list[ViewDescriptor] = []
    full_indices = tuple(range(split.X_train.shape[1]))
    family_overrides = dict(family_overrides or {})

    for name in views.view_names:
        if name == "FULL":
            descriptors.append(
                ViewDescriptor(
                    expert_id="FULL",
                    family="FULL",
                    view_name="FULL",
                    projection_kind="identity_subselect",
                    input_dim=len(full_indices),
                    input_indices=full_indices,
                    feature_names=split.feature_names,
                    is_anchor=True,
                    source_name=split.dataset_key,
                    tags=("anchor", "portfolio"),
                )
            )
            continue

        family = family_overrides.get(name, _default_family_for_view(name))
        if name == "LOWRANK":
            descriptors.append(
                ViewDescriptor(
                    expert_id="LOWRANK",
                    family=family,
                    view_name="LOWRANK",
                    projection_kind="external_transform",
                    input_dim=views.train[name].shape[1],
                    feature_names=tuple(f"lowrank_{idx}" for idx in range(views.train[name].shape[1])),
                    source_name=split.dataset_key,
                    tags=("portfolio", "transformed"),
                )
            )
            continue

        input_indices = split.view_columns[name]
        feature_names = tuple(split.feature_names[idx] for idx in input_indices)
        descriptors.append(
            ViewDescriptor(
                expert_id=name,
                family=family,
                view_name=name,
                projection_kind="identity_subselect",
                input_dim=len(input_indices),
                input_indices=input_indices,
                feature_names=feature_names,
                source_name=split.dataset_key,
                tags=("portfolio", "registry"),
            )
        )

    return GraphDroneBenchmarkDescriptorSet(
        dataset_key=split.dataset_key,
        descriptors=normalize_descriptor_set(descriptors, required_anchor_id="FULL"),
    )


def _default_family_for_view(view_name: str) -> str:
    if view_name == "FULL":
        return "FULL"
    if view_name == "LOWRANK":
        return "structural_subspace"
    return "bootstrap"
