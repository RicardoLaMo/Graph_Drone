from __future__ import annotations

from dataclasses import dataclass


VALID_EXPERT_FAMILIES = frozenset(
    {
        "FULL",
        "anchor_generalist",
        "feature_subset",
        "structural_subspace",
        "local_support",
        "learned_regime",
        "domain_semantic",
        "geometry_signal",
        "numeric_subset",
        "categorical_subset",
        "bootstrap",
    }
)
VALID_PROJECTION_KINDS = frozenset(
    {
        "identity_subselect",
        "external_transform",
        "derived_features",
        "support_defined",
        "opaque",
    }
)


@dataclass(frozen=True)
class ViewDescriptor:
    expert_id: str
    family: str
    view_name: str
    projection_kind: str = "identity_subselect"
    input_dim: int = 0
    input_indices: tuple[int, ...] = ()
    feature_names: tuple[str, ...] = ()
    is_anchor: bool = False
    source_name: str = ""
    tags: tuple[str, ...] = ()

    def validate(self) -> "ViewDescriptor":
        if not self.expert_id.strip():
            raise ValueError("expert_id must be non-empty")
        if not self.view_name.strip():
            raise ValueError("view_name must be non-empty")
        if self.family not in VALID_EXPERT_FAMILIES:
            raise ValueError(f"Unsupported family={self.family!r}")
        if self.projection_kind not in VALID_PROJECTION_KINDS:
            raise ValueError(f"Unsupported projection_kind={self.projection_kind!r}")
        if self.input_dim < 0:
            raise ValueError(f"input_dim must be non-negative, got {self.input_dim}")
        if len(set(self.input_indices)) != len(self.input_indices):
            raise ValueError(f"input_indices must be unique for expert_id={self.expert_id!r}")
        if any(idx < 0 for idx in self.input_indices):
            raise ValueError(f"input_indices must be non-negative for expert_id={self.expert_id!r}")
        if self.feature_names and self.input_dim and len(self.feature_names) != self.input_dim:
            raise ValueError(
                f"feature_names length {len(self.feature_names)} does not match input_dim={self.input_dim}"
            )
        if self.projection_kind == "identity_subselect" and self.input_dim != len(self.input_indices):
            raise ValueError(
                "identity_subselect descriptors must provide one input index per input dimension"
            )
        if self.is_anchor and self.family not in {"FULL", "anchor_generalist"}:
            raise ValueError("Only anchor-capable expert families may be marked as anchors")
        return self

    def to_dict(self) -> dict[str, object]:
        return {
            "expert_id": self.expert_id,
            "family": self.family,
            "view_name": self.view_name,
            "projection_kind": self.projection_kind,
            "input_dim": self.input_dim,
            "input_indices": list(self.input_indices),
            "feature_names": list(self.feature_names),
            "is_anchor": self.is_anchor,
            "source_name": self.source_name,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ViewDescriptor":
        return cls(
            expert_id=str(payload["expert_id"]),
            family=str(payload["family"]),
            view_name=str(payload["view_name"]),
            projection_kind=str(payload.get("projection_kind", "identity_subselect")),
            input_dim=int(payload.get("input_dim", 0)),
            input_indices=tuple(int(v) for v in payload.get("input_indices", [])),
            feature_names=tuple(str(v) for v in payload.get("feature_names", [])),
            is_anchor=bool(payload.get("is_anchor", False)),
            source_name=str(payload.get("source_name", "")),
            tags=tuple(str(v) for v in payload.get("tags", [])),
        ).validate()


def normalize_descriptor_set(
    descriptors: list[ViewDescriptor] | tuple[ViewDescriptor, ...],
    *,
    required_anchor_id: str = "FULL",
) -> tuple[ViewDescriptor, ...]:
    normalized = tuple(descriptor.validate() for descriptor in descriptors)
    ids = [descriptor.expert_id for descriptor in normalized]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate expert ids found: {ids!r}")
    anchors = [descriptor.expert_id for descriptor in normalized if descriptor.is_anchor]
    if not anchors:
        raise ValueError("At least one anchor descriptor is required")
    if required_anchor_id not in ids:
        raise ValueError(f"Required anchor expert_id={required_anchor_id!r} is missing")
    return normalized
