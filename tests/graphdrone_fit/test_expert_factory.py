from __future__ import annotations

import numpy as np

from src.graphdrone_fit.expert_factory import (
    ExpertBuildSpec,
    IdentitySelectorAdapter,
    PcaProjectionAdapter,
    fit_portfolio_from_specs,
)
from src.graphdrone_fit.view_descriptor import ViewDescriptor


def test_fit_portfolio_from_specs_builds_predictable_linear_portfolio() -> None:
    X_train = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.0],
            [0.5, 0.5, 1.5],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0.0, 1.0, 0.5], dtype=np.float32)
    specs = (
        ExpertBuildSpec(
            descriptor=ViewDescriptor(
                expert_id="FULL",
                family="FULL",
                view_name="FULL",
                projection_kind="identity_subselect",
                input_dim=3,
                input_indices=(0, 1, 2),
                is_anchor=True,
            ),
            model_kind="linear",
            input_adapter=IdentitySelectorAdapter(indices=(0, 1, 2)),
        ),
        ExpertBuildSpec(
            descriptor=ViewDescriptor(
                expert_id="LOCAL",
                family="local_support",
                view_name="LOCAL",
                projection_kind="identity_subselect",
                input_dim=2,
                input_indices=(0, 2),
            ),
            model_kind="constant",
            input_adapter=IdentitySelectorAdapter(indices=(0, 2)),
            model_params={"value": 0.75},
        ),
    )

    portfolio = fit_portfolio_from_specs(
        X_train=X_train,
        y_train=y_train,
        specs=specs,
        full_expert_id="FULL",
    )
    X_test = np.array([[3.0, 1.0, 2.0]], dtype=np.float32)
    full_pred = portfolio.experts["FULL"].predict(X_test)
    local_pred = portfolio.experts["LOCAL"].predict(X_test)
    assert full_pred.shape == (1,)
    assert local_pred.shape == (1,)
    assert np.isclose(local_pred[0], 0.75)


def test_pca_projection_adapter_requires_fit_before_transform() -> None:
    adapter = PcaProjectionAdapter(n_components=2, random_state=7)
    X = np.array([[1.0, 2.0], [2.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    try:
        adapter.transform(X)
    except RuntimeError as exc:
        assert "fit" in str(exc)
    else:
        raise AssertionError("Expected transform() to require prior fit()")

    fitted = adapter.fit(X)
    projected = fitted.transform(X)
    assert projected.shape == (3, 2)


def test_fit_portfolio_from_specs_rejects_unsupported_model_kind() -> None:
    X_train = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)
    y_train = np.array([0.0, 1.0], dtype=np.float32)
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
            model_kind="does_not_exist",
            input_adapter=IdentitySelectorAdapter(indices=(0, 1)),
        ),
    )

    try:
        fit_portfolio_from_specs(
            X_train=X_train,
            y_train=y_train,
            specs=specs,
            full_expert_id="FULL",
        )
    except ValueError as exc:
        assert "Unsupported model_kind" in str(exc)
    else:
        raise AssertionError("Expected unsupported model_kind to raise ValueError")
