"""
Unit tests for task_conditioned_prior and TaskConditionedRouter (Phase 2 / V1.3).

Tests the bank serialization round-trip, encoder forward pass,
TaskConditionedRouter passthrough and injection modes.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from graphdrone_fit.task_conditioned_prior import (
    TaskContextBatch,
    TaskContextNormalization,
    TaskPrototypeBank,
    apply_task_context_normalization,
    build_task_context_encoder,
    fit_task_context_normalization,
    load_task_prototype_bank,
    normalized_centroids,
    save_task_prototype_bank,
    supervised_contrastive_loss,
)
from graphdrone_fit.set_router import TaskConditionedRouter


# ---------------------------------------------------------------------------
# Bank serialization round-trip
# ---------------------------------------------------------------------------

def _make_dummy_bank(dataset_names: tuple[str, ...] = ("ds_a", "ds_b")) -> TaskPrototypeBank:
    dim = 8
    centroids = {name: F.normalize(torch.randn(dim), dim=-1) for name in dataset_names}
    counts = {name: 3 for name in dataset_names}
    return TaskPrototypeBank(
        dataset_names=dataset_names,
        centroids=centroids,
        counts=counts,
        feature_names=tuple(f"f{i}" for i in range(4)),
        encoder_kind="transformer",
        hidden_dim=dim,
        normalize_features=True,
    )


def test_bank_serialization_round_trip():
    bank = _make_dummy_bank()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bank.json"
        save_task_prototype_bank(bank, path)
        assert path.exists()
        loaded = load_task_prototype_bank(path)

    assert loaded.dataset_names == bank.dataset_names
    assert loaded.encoder_kind == bank.encoder_kind
    assert loaded.hidden_dim == bank.hidden_dim
    assert loaded.normalize_features == bank.normalize_features
    for name in bank.dataset_names:
        assert name in loaded.centroids
        torch.testing.assert_close(loaded.centroids[name], bank.centroids[name], atol=1e-5, rtol=0)


def test_bank_serialization_with_normalization():
    bank = _make_dummy_bank()
    norm = TaskContextNormalization(
        mean=torch.zeros(6),
        std=torch.ones(6),
        continuous_mask=torch.tensor([True, True, False, True, False, True]),
    )
    bank_with_norm = TaskPrototypeBank(
        dataset_names=bank.dataset_names,
        centroids=bank.centroids,
        counts=bank.counts,
        feature_names=bank.feature_names,
        encoder_kind=bank.encoder_kind,
        hidden_dim=bank.hidden_dim,
        normalize_features=bank.normalize_features,
        normalization=norm,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bank_norm.json"
        save_task_prototype_bank(bank_with_norm, path)
        loaded = load_task_prototype_bank(path)
    assert loaded.normalization is not None
    torch.testing.assert_close(loaded.normalization.mean, norm.mean)
    torch.testing.assert_close(loaded.normalization.std, norm.std)


# ---------------------------------------------------------------------------
# Encoder forward pass
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("encoder_kind", ["transformer", "gru"])
def test_encoder_forward_shape(encoder_kind: str):
    input_dim, hidden_dim, batch, seq_len = 10, 32, 4, 3
    encoder = build_task_context_encoder(encoder_kind, input_dim=input_dim, hidden_dim=hidden_dim)
    sequences = torch.randn(batch, seq_len, input_dim)
    with torch.no_grad():
        out = encoder(sequences)
    assert out.shape == (batch, hidden_dim), f"expected ({batch}, {hidden_dim}), got {tuple(out.shape)}"


def test_encoder_unknown_kind_raises():
    with pytest.raises(ValueError, match="Unsupported encoder_kind"):
        build_task_context_encoder("unknown_kind", input_dim=10, hidden_dim=32)


# ---------------------------------------------------------------------------
# TaskConditionedRouter
# ---------------------------------------------------------------------------

class _DummyRouter(torch.nn.Module):
    """Minimal router for testing TaskConditionedRouter wrapper."""

    router_kind = "dummy"

    def __init__(self, token_dim: int, n_experts: int):
        super().__init__()
        self.token_dim = token_dim
        self.n_experts = n_experts

    def forward(self, tokens: torch.Tensor, *, full_index: int):
        from graphdrone_fit.set_router import RouterOutputs
        B = tokens.shape[0]
        E = tokens.shape[1]
        weights = torch.ones(B, E - 1) / (E - 1)  # uniform specialist weights
        defer_prob = torch.full((B, 1), 0.5)
        return RouterOutputs(
            specialist_weights=weights,
            defer_prob=defer_prob,
            full_index=full_index,
            router_kind="dummy",
            extra_diagnostics={},
        )


def test_task_conditioned_router_passthrough_without_prior():
    token_dim, prior_dim = 16, 8
    base = _DummyRouter(token_dim=token_dim, n_experts=3)
    router = TaskConditionedRouter(
        token_dim=token_dim,
        prior_dim=prior_dim,
        base_router=base,
        strength=1.0,
        router_kind="dummy_task_prior",
    )
    tokens = torch.randn(5, 3, token_dim)
    out = router(tokens, full_index=0)
    # No prior set → should passthrough with task_prior_enabled=0
    assert out.extra_diagnostics is not None
    assert out.extra_diagnostics.get("task_prior_enabled") == 0.0


def test_task_conditioned_router_with_prior():
    token_dim, prior_dim = 16, 8
    base = _DummyRouter(token_dim=token_dim, n_experts=3)
    router = TaskConditionedRouter(
        token_dim=token_dim,
        prior_dim=prior_dim,
        base_router=base,
        strength=1.0,
        router_kind="dummy_task_prior",
    )
    prior_ctx = torch.randn(prior_dim)
    router.set_task_prior_context(prior_ctx)
    tokens = torch.randn(5, 3, token_dim)
    out = router(tokens, full_index=0)
    assert out.extra_diagnostics is not None
    assert out.extra_diagnostics.get("task_prior_enabled") == 1.0
    assert "task_prior_strength" in out.extra_diagnostics
    assert "task_prior_norm" in out.extra_diagnostics


def test_task_conditioned_router_strength_zero_passthrough():
    token_dim, prior_dim = 16, 8
    base = _DummyRouter(token_dim=token_dim, n_experts=3)
    router = TaskConditionedRouter(
        token_dim=token_dim,
        prior_dim=prior_dim,
        base_router=base,
        strength=0.0,
        router_kind="dummy_task_prior",
    )
    router.set_task_prior_context(torch.randn(prior_dim))
    tokens = torch.randn(5, 3, token_dim)
    out = router(tokens, full_index=0)
    # strength=0 → passthrough even if prior is set
    assert out.extra_diagnostics is not None
    assert out.extra_diagnostics.get("task_prior_enabled") == 0.0


def test_task_conditioned_router_dim_mismatch_raises():
    token_dim, prior_dim = 16, 8
    base = _DummyRouter(token_dim=token_dim, n_experts=3)
    router = TaskConditionedRouter(
        token_dim=token_dim,
        prior_dim=prior_dim,
        base_router=base,
        strength=1.0,
        router_kind="dummy_task_prior",
    )
    wrong_dim_ctx = torch.randn(prior_dim + 4)
    with pytest.raises(ValueError, match="dim mismatch"):
        router.set_task_prior_context(wrong_dim_ctx)


# ---------------------------------------------------------------------------
# Supervised contrastive loss
# ---------------------------------------------------------------------------

def test_supervised_contrastive_loss_positive():
    torch.manual_seed(0)
    embeddings = torch.randn(8, 16)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 2, 2])
    loss = supervised_contrastive_loss(embeddings, labels)
    assert loss.item() > 0


def test_supervised_contrastive_loss_requires_positive_pairs():
    embeddings = torch.randn(3, 8)
    labels = torch.tensor([0, 1, 2])  # all unique → no positive pairs
    with pytest.raises(ValueError, match="positive pair"):
        supervised_contrastive_loss(embeddings, labels)
