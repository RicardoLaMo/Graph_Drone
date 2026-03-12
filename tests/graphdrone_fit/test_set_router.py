from __future__ import annotations

import numpy as np
import torch

from src.graphdrone_fit.config import SetRouterConfig
from src.graphdrone_fit.set_router import ContextualSparseRouter, build_set_router


def test_contextual_sparse_router_learns_row_level_defer_signal() -> None:
    rng = np.random.default_rng(7)
    x = rng.uniform(-1.0, 1.0, size=(160, 1)).astype(np.float32)
    full_pred = np.zeros((len(x), 1), dtype=np.float32)
    specialist_pred = x.copy()
    predictions = np.concatenate([full_pred, specialist_pred], axis=1)

    quality = np.zeros((len(x), 2, 2), dtype=np.float32)
    quality[:, 0, 0] = 0.0
    quality[:, 1, 0] = np.clip(x[:, 0], 0.0, None)
    quality[:, 0, 1] = 1.0
    quality[:, 1, 1] = 0.0

    tokens = np.concatenate(
        [
            np.stack(
                [
                    predictions,
                    predictions - predictions[:, :1],
                    predictions - predictions.mean(axis=1, keepdims=True),
                ],
                axis=-1,
            ),
            quality,
        ],
        axis=-1,
    )
    y = np.clip(x[:, 0], 0.0, None).astype(np.float32)

    router = ContextualSparseRouter(
        SetRouterConfig(
            kind="contextual_sparse_mlp",
            sparse_top_k=1,
            hidden_dim=16,
            learning_rate=5e-2,
            weight_decay=0.0,
            max_epochs=120,
            patience=15,
            validation_fraction=0.2,
            random_seed=7,
        )
    )
    summary = router.fit_router(
        torch.as_tensor(tokens),
        torch.as_tensor(predictions),
        torch.as_tensor(y),
        full_index=0,
        field_slices={"prediction": (0, 3), "quality": (3, 5)},
    )
    outputs = router(
        torch.as_tensor(tokens),
        full_index=0,
        field_slices={"prediction": (0, 3), "quality": (3, 5)},
    )
    defer = outputs.defer_prob.squeeze(1).detach().cpu().numpy()
    specialist_weights = outputs.specialist_weights.detach().cpu().numpy()

    assert summary["fit_status"] == "fitted"
    assert summary["token_encoder_kind"] == "field_aware"
    assert specialist_weights[:, 0].max() == 0.0
    assert defer[x[:, 0] > 0.5].mean() > defer[x[:, 0] < -0.5].mean()
    assert summary["mean_defer_prob"] > 0.0


def test_build_set_router_supports_contextual_kind() -> None:
    router = build_set_router(SetRouterConfig(kind="contextual_sparse_mlp"))
    assert isinstance(router, ContextualSparseRouter)


def test_contextual_sparse_router_can_force_flat_encoder_even_with_field_slices() -> None:
    tokens = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
            [[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]],
            [[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]],
        ],
        dtype=torch.float32,
    )
    predictions = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
            [0.0, 4.0],
        ],
        dtype=torch.float32,
    )
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    router = ContextualSparseRouter(
        SetRouterConfig(
            kind="contextual_sparse_mlp",
            field_aware_tokens=False,
            hidden_dim=8,
            learning_rate=5e-2,
            weight_decay=0.0,
            max_epochs=40,
            patience=5,
            validation_fraction=0.25,
            random_seed=11,
        )
    )
    field_slices = {"prediction": (0, 2), "quality": (2, 3)}
    summary = router.fit_router(tokens, predictions, y_true, full_index=0, field_slices=field_slices)
    outputs = router(tokens, full_index=0, field_slices=field_slices)

    assert summary["fit_status"] == "fitted"
    assert summary["token_encoder_kind"] == "flat_mlp"
    assert outputs.specialist_weights.shape == predictions.shape
