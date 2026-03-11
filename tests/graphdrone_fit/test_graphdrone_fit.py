from __future__ import annotations

import json

import numpy as np

from src.graphdrone_fit import (
    ExpertBuildSpec,
    GraphDrone,
    GraphDroneConfig,
    IdentitySelectorAdapter,
    PortfolioLoadConfig,
    SetRouterConfig,
    ViewDescriptor,
)
from src.graphdrone_fit.portfolio_loader import load_portfolio
from src.graphdrone_fit.token_builder import PerViewTokenBuilder, build_legacy_quality_encoding_from_flat
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
    quality = build_legacy_quality_encoding_from_flat(
        view_names=["FULL", "SUPPORT"],
        flat_quality=np.array([[0.1, 0.2, 0.4, 0.7], [0.2, 0.1, 0.3, 0.6]], dtype=np.float32),
    )
    result = model.predict(X_test, quality_features=quality, return_diagnostics=True)
    expected = np.array([3.0 - 2.0 + 0.25, 0.0 - 1.0 + 0.25], dtype=np.float32)
    assert np.allclose(result.predictions, expected)
    assert result.diagnostics["router_kind"] == "bootstrap_full_only"
    assert result.diagnostics["full_expert_id"] == "FULL"
    assert result.diagnostics["quality_feature_names"][0] == "quality_sigma2_self"
    assert "descriptor_is_anchor" in result.diagnostics["token_field_names"]["descriptor"]
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


def test_graphdrone_fit_accepts_real_expert_specs_without_manifest() -> None:
    X_train = np.array([[1.0, 2.0], [2.0, 0.5], [0.0, 1.5]], dtype=np.float32)
    y_train = np.array([1.0, 0.0, 0.5], dtype=np.float32)
    config = GraphDroneConfig(
        portfolio=None,
        full_expert_id="FULL",
        router=SetRouterConfig(kind="bootstrap_full_only"),
    )
    model = GraphDrone(config)
    specs = (
        ExpertBuildSpec(
            descriptor=ViewDescriptor(
                expert_id="FULL",
                family="FULL",
                view_name="FULL",
                projection_kind="identity_subselect",
                input_dim=2,
                input_indices=(0, 1),
                is_anchor=True,
            ),
            model_kind="linear",
            input_adapter=IdentitySelectorAdapter(indices=(0, 1)),
        ),
        ExpertBuildSpec(
            descriptor=ViewDescriptor(
                expert_id="SPECIALIST",
                family="local_support",
                view_name="SPECIALIST",
                projection_kind="identity_subselect",
                input_dim=1,
                input_indices=(1,),
            ),
            model_kind="constant",
            input_adapter=IdentitySelectorAdapter(indices=(1,)),
            model_params={"value": 0.25},
        ),
    )

    model.fit(X_train, y_train, expert_specs=specs)
    result = model.predict(X_train, return_diagnostics=True)
    assert result.diagnostics["full_expert_id"] == "FULL"
    assert result.token_shape[1] == 2


def test_graphdrone_fit_requires_y_for_expert_specs() -> None:
    X_train = np.array([[1.0, 2.0], [2.0, 0.5]], dtype=np.float32)
    model = GraphDrone(
        GraphDroneConfig(
            portfolio=None,
            full_expert_id="FULL",
            router=SetRouterConfig(kind="bootstrap_full_only"),
        )
    )
    specs = (
        ExpertBuildSpec(
            descriptor=ViewDescriptor(
                expert_id="FULL",
                family="FULL",
                view_name="FULL",
                projection_kind="identity_subselect",
                input_dim=2,
                input_indices=(0, 1),
                is_anchor=True,
            ),
            model_kind="linear",
            input_adapter=IdentitySelectorAdapter(indices=(0, 1)),
        ),
    )

    try:
        model.fit(X_train, expert_specs=specs)
    except ValueError as exc:
        assert "y is required" in str(exc)
    else:
        raise AssertionError("Expected GraphDrone.fit() to require y when expert_specs are provided")


def test_graphdrone_predict_exposes_active_specialist_ids_after_router_fit() -> None:
    X_train = np.array(
        [
            [-1.0, 0.0],
            [-0.5, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [1.5, 0.0],
        ],
        dtype=np.float32,
    )
    y_train = np.clip(X_train[:, 0], 0.0, None).astype(np.float32)
    model = GraphDrone(
        GraphDroneConfig(
            portfolio=None,
            full_expert_id="FULL",
            router=SetRouterConfig(
                kind="contextual_sparse_mlp",
                hidden_dim=8,
                learning_rate=5e-2,
                weight_decay=0.0,
                max_epochs=40,
                patience=5,
                validation_fraction=0.4,
                random_seed=7,
                sparse_top_k=1,
            ),
        )
    )
    specs = (
        ExpertBuildSpec(
            descriptor=ViewDescriptor(
                expert_id="FULL",
                family="FULL",
                view_name="FULL",
                projection_kind="identity_subselect",
                input_dim=2,
                input_indices=(0, 1),
                is_anchor=True,
            ),
            model_kind="constant",
            input_adapter=IdentitySelectorAdapter(indices=(0, 1)),
            model_params={"value": 0.0},
        ),
        ExpertBuildSpec(
            descriptor=ViewDescriptor(
                expert_id="SPECIALIST",
                family="local_support",
                view_name="SPECIALIST",
                projection_kind="identity_subselect",
                input_dim=1,
                input_indices=(0,),
            ),
            model_kind="linear",
            input_adapter=IdentitySelectorAdapter(indices=(0,)),
        ),
    )
    model.fit(X_train, y_train, expert_specs=specs)
    quality = np.zeros((len(X_train), 2, 2), dtype=np.float32)
    quality[:, 1, 0] = np.clip(X_train[:, 0], 0.0, None)
    quality[:, 1, 1] = 1.0
    model.fit_router(X_train, y_train, quality_features=quality)
    result = model.predict(X_train, quality_features=quality, return_diagnostics=True)
    summary = result.diagnostics["router_fit_summary"]
    assert "active_specialist_ids" in summary
    assert isinstance(summary["active_specialist_ids"], list)
    if summary["active_specialist_ids"]:
        assert summary["active_specialist_ids"] == ["SPECIALIST"]


def test_graphdrone_predict_records_support_summary_fields_from_4d_support_tensor(tmp_path) -> None:
    _write_manifest(tmp_path)
    model = GraphDrone(
        GraphDroneConfig(
            portfolio=PortfolioLoadConfig(manifest_path=tmp_path / "portfolio_manifest.json"),
            full_expert_id="FULL",
            router=SetRouterConfig(kind="bootstrap_full_only"),
        )
    )
    model.fit(np.array([[1.0, 2.0], [2.0, 0.5]], dtype=np.float32))
    X_test = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    support = np.array(
        [
            [
                [[1.0], [2.0], [3.0]],
                [[0.5], [1.5], [2.5]],
            ],
            [
                [[0.0], [1.0], [2.0]],
                [[1.0], [2.0], [3.0]],
            ],
        ],
        dtype=np.float32,
    )

    result = model.predict(X_test, support_tensor=support, return_diagnostics=True)
    assert result.diagnostics["support_feature_names"] == [
        "support_mean_0",
        "support_std_0",
        "support_absmax_0",
        "support_count",
    ]
    assert result.diagnostics["token_field_slices"]["support"] == [3, 7]


def test_graphdrone_fit_router_records_summary_on_contextual_router() -> None:
    X_train = np.array(
        [[-1.0], [-0.5], [0.0], [0.5], [1.0], [1.5]],
        dtype=np.float32,
    )
    y_train = X_train[:, 0].copy()
    X_router = np.array([[-1.0], [-0.5], [0.5], [1.0]], dtype=np.float32)
    y_router = np.clip(X_router[:, 0], 0.0, None)
    model = GraphDrone(
        GraphDroneConfig(
            portfolio=None,
            full_expert_id="FULL",
            router=SetRouterConfig(
                kind="contextual_sparse_mlp",
                sparse_top_k=1,
                hidden_dim=16,
                learning_rate=5e-2,
                weight_decay=0.0,
                max_epochs=80,
                patience=10,
                validation_fraction=0.25,
                random_seed=3,
            ),
        )
    )
    specs = (
        ExpertBuildSpec(
            descriptor=ViewDescriptor(
                expert_id="FULL",
                family="FULL",
                view_name="FULL",
                projection_kind="identity_subselect",
                input_dim=1,
                input_indices=(0,),
                is_anchor=True,
            ),
            model_kind="constant",
            input_adapter=IdentitySelectorAdapter(indices=(0,)),
            model_params={"value": 0.0},
        ),
        ExpertBuildSpec(
            descriptor=ViewDescriptor(
                expert_id="SPECIALIST",
                family="local_support",
                view_name="SPECIALIST",
                projection_kind="identity_subselect",
                input_dim=1,
                input_indices=(0,),
            ),
            model_kind="linear",
            input_adapter=IdentitySelectorAdapter(indices=(0,)),
        ),
    )
    model.fit(X_train, y_train, expert_specs=specs)
    summary = model.fit_router(X_router, y_router)
    result = model.predict(X_router, return_diagnostics=True)

    assert summary["fit_status"] == "fitted"
    assert result.diagnostics["router_fit_summary"]["fit_status"] == "fitted"
    assert result.diagnostics["router_kind"] == "contextual_sparse_mlp"


def test_graphdrone_classification_bootstrap_predicts_probabilities() -> None:
    X_train = np.array(
        [
            [0.0, 0.1],
            [0.2, 0.0],
            [1.0, 1.1],
            [1.2, 1.0],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 1, 1], dtype=np.int64)
    model = GraphDrone(
        GraphDroneConfig(
            portfolio=None,
            full_expert_id="FULL",
            task_type="classification",
            router=SetRouterConfig(kind="bootstrap_full_only"),
        )
    )
    specs = (
        ExpertBuildSpec(
            descriptor=ViewDescriptor(
                expert_id="FULL",
                family="FULL",
                view_name="FULL",
                projection_kind="identity_subselect",
                input_dim=2,
                input_indices=(0, 1),
                is_anchor=True,
            ),
            model_kind="logistic_classifier",
            input_adapter=IdentitySelectorAdapter(indices=(0, 1)),
        ),
        ExpertBuildSpec(
            descriptor=ViewDescriptor(
                expert_id="SPECIALIST",
                family="bootstrap",
                view_name="SPECIALIST",
                projection_kind="identity_subselect",
                input_dim=1,
                input_indices=(1,),
            ),
            model_kind="constant_classifier",
            input_adapter=IdentitySelectorAdapter(indices=(1,)),
        ),
    )

    model.fit(X_train, y_train, expert_specs=specs)
    probabilities = model.predict_proba(X_train)
    labels = model.predict(X_train)
    result = model.predict(X_train, return_diagnostics=True)

    assert probabilities.shape == (4, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert labels.shape == (4,)
    assert result.probabilities is not None
    assert result.class_labels == (0, 1)
    assert result.diagnostics["task_type"] == "classification"


def test_token_builder_emits_classification_tensor_fields() -> None:
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
                expert_id="SPECIALIST",
                family="bootstrap",
                view_name="SPECIALIST",
                projection_kind="identity_subselect",
                input_dim=1,
                input_indices=(1,),
            ),
        ],
        required_anchor_id="FULL",
    )
    probabilities = np.array(
        [
            [[0.8, 0.2], [0.6, 0.4]],
            [[0.3, 0.7], [0.4, 0.6]],
        ],
        dtype=np.float32,
    )
    batch = PerViewTokenBuilder().build(
        predictions=probabilities,
        descriptors=descriptors,
        full_expert_id="FULL",
    )
    assert batch.tokens.shape[0] == 2
    assert batch.tokens.shape[1] == 2
    assert batch.field_slices["prediction"] == (0, 11)
    assert "prediction_proba_class_0" in batch.field_names["prediction"]
    assert "prediction_entropy_minus_full" in batch.field_names["prediction"]
