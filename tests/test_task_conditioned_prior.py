from __future__ import annotations

import json

import pandas as pd
import torch

from graphdrone_fit.task_conditioned_prior import (
    TaskContextGRUEncoder,
    TaskContextTransformerEncoder,
    build_task_context_batch,
)


def _task_context_frame() -> pd.DataFrame:
    rows = []
    for dataset in ["a", "b"]:
        for bootstrap_id in [0, 1]:
            for expert_id, family, is_anchor in [("FULL", "FULL", 1), ("SUB0", "structural_subspace", 0), ("SUB1", "structural_subspace", 0)]:
                rows.append(
                    {
                        "dataset": dataset,
                        "task_type": "classification",
                        "bootstrap_id": bootstrap_id,
                        "expert_id": expert_id,
                        "family": family,
                        "projection_kind": "identity_subselect",
                        "input_dim": 8 if is_anchor else 4,
                        "preferred_k": 15,
                        "is_anchor": is_anchor,
                        "mean_norm": 1.0,
                        "std_norm": 0.5,
                        "mean_token_json": json.dumps([0.1, 0.2, 0.3]),
                    }
                )
    return pd.DataFrame(rows)


def test_build_task_context_batch_orders_full_first() -> None:
    batch = build_task_context_batch(_task_context_frame())
    assert batch.sequences.shape == (4, 3, batch.sequences.shape[-1])
    assert batch.labels.shape[0] == 4
    assert batch.dataset_names == ("a", "b")


def test_transformer_encoder_forward_shape() -> None:
    batch = build_task_context_batch(_task_context_frame())
    encoder = TaskContextTransformerEncoder(input_dim=batch.sequences.shape[-1], hidden_dim=32, num_heads=4, num_layers=1)
    out = encoder(batch.sequences)
    assert out.shape == (4, 32)
    assert torch.isfinite(out).all()


def test_gru_encoder_forward_shape() -> None:
    batch = build_task_context_batch(_task_context_frame())
    encoder = TaskContextGRUEncoder(input_dim=batch.sequences.shape[-1], hidden_dim=32, num_layers=1)
    out = encoder(batch.sequences)
    assert out.shape == (4, 32)
    assert torch.isfinite(out).all()
