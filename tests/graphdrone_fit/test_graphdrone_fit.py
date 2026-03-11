from __future__ import annotations

import json

import numpy as np

from src.graphdrone_fit import GraphDrone, GraphDroneConfig, PortfolioLoadConfig, SetRouterConfig, ViewDescriptor
from src.graphdrone_fit.portfolio_loader import load_portfolio
from src.graphdrone_fit.token_builder import PerViewTokenBuilder
from src.graphdrone_fit.view_descriptor import normalize_descriptor_set


def _write_manifest(tmp_path) -> None:
    payload = {
        "full_expert_id": "FULL",
        "experts": [
            {
                "descriptor": ViewDescriptor(
                    expert_id="FULL",
                    family="FULL",
                    view_name="FULL",
                    projection_kind="identity_subselect",
                    input_dim=2,
                    input_indices=(0, 1),
                    feature_names=("f0", "f1"),
                    is_anchor=True,
                    source_name="standalone",
                    tags=("anchor",),
                ).to_dict(),
                "artifact": {
                    "kind": "linear",
                    "coefficients": [1.0, -0.5],
                    "bias": 0.25,
                },
            },
            {
                "descriptor": ViewDescriptor(
                    expert_id="SUPPORT",
                    family="local_support",
                    view_name="SUPPORT",
                    projection_kind="identity_subselect",
                    input_dim=1,
                    input_indices=(1,),
                    feature_names=("f1",),
                    source_name="standalone",
                    tags=("specialist",),
                ).to_dict(),
                "artifact": {
                    "kind": "constant",
                    "value": 1.5,
                },
            },
        ],
    }
    (tmp_path / "portfolio_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def test_normalize_descriptor_set_requires_anchor() -> None:
    descriptors = [
        ViewDescriptor(
            expert_id="A",
            family="local_support",
            view_name="A",
            projection_kind="identity_subselect",
            input_dim=1,
            input_indices=(0,),
        )
    ]
    try:
        normalize_descriptor_set(descriptors, required_anchor_id="FULL")
    except ValueError as exc:
        assert "anchor" in str(exc)
    else:
        raise AssertionError("Expected normalize_descriptor_set to reject missing anchor")


def test_portfolio_loader_loads_manifest_without_registry(tmp_path) -> None:
    _write_manifest(tmp_path)
    portfolio = load_portfolio(PortfolioLoadConfig(manifest_path=tmp_path / "portfolio_manifest.json"))
    assert portfolio.full_expert_id == "FULL"
    assert portfolio.expert_order == ("FULL", "SUPPORT")
    assert portfolio.manifest_path == (tmp_path / "portfolio_manifest.json").resolve()


def test_graphdrone_bootstrap_predict_uses_explicit_full_anchor(tmp_path) -> None:
    _write_manifest(tmp_path)
    config = GraphDroneConfig(
        portfolio=PortfolioLoadConfig(manifest_path=tmp_path / "portfolio_manifest.json"),
        full_expert_id="FULL",
        router=SetRouterConfig(kind="bootstrap_full_only"),
    )
    model = GraphDrone(config)
    X_train = np.array([[1.0, 2.0], [2.0, 0.5]], dtype=np.float32)
    model.fit(X_train)

    X_test = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    result = model.predict(X_test, return_diagnostics=True)
    expected = np.array([3.0 - 2.0 + 0.25, 0.0 - 1.0 + 0.25], dtype=np.float32)
    assert np.allclose(result.predictions, expected)
    assert result.diagnostics["router_kind"] == "bootstrap_full_only"
    assert result.diagnostics["full_expert_id"] == "FULL"
    assert result.token_shape[0] == len(X_test)
    assert result.token_shape[1] == 2


def test_token_builder_emits_tensor_fields() -> None:
    descriptors = normalize_descriptor_set(
        [
            ViewDescriptor(
                expert_id="FULL",
                family="FULL",
                view_name="FULL",
                projection_kind="identity_subselect",
                input_dim=2,
                input_indices=(0, 1),
                is_anchor=True,
            ),
            ViewDescriptor(
                expert_id="LOWRANK",
                family="structural_subspace",
                view_name="LOWRANK",
                projection_kind="external_transform",
                input_dim=1,
            ),
        ],
        required_anchor_id="FULL",
    )
    predictions = np.array([[1.0, 1.5], [2.0, 1.0]], dtype=np.float32)
    quality = np.array(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
        ],
        dtype=np.float32,
    )
    batch = PerViewTokenBuilder().build(
        predictions=predictions,
        descriptors=descriptors,
        full_expert_id="FULL",
        quality_features=quality,
    )
    assert batch.tokens.shape[0] == 2
    assert batch.tokens.shape[1] == 2
    assert batch.field_slices["prediction"] == (0, 3)
    assert batch.field_slices["quality"] == (3, 5)
