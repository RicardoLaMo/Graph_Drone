from __future__ import annotations

import json

import pandas as pd
import torch

from graphdrone_fit.task_conditioned_prior import (
    TaskContextSequenceAutoencoder,
    TaskContextGRUEncoder,
    apply_task_context_normalization,
    build_task_prototype_bank,
    embedding_neighbor_distribution,
    fit_task_context_normalization,
    metadata_neighbor_targets,
    neighborhood_consistency_loss,
    TaskContextTransformerEncoder,
    build_task_context_batch,
    load_task_prototype_bank,
    query_task_prototype_bank,
    save_task_prototype_bank,
    slice_batch_by_datasets,
    split_batch_by_dataset,
    supervised_contrastive_loss,
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
    assert batch.example_datasets == ("a", "a", "b", "b")


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


def test_split_batch_by_dataset() -> None:
    batch = build_task_context_batch(_task_context_frame())
    train_batch, test_batch = split_batch_by_dataset(batch, "b")
    assert train_batch.sequences.shape[0] == 2
    assert test_batch.sequences.shape[0] == 2
    assert all(name == "a" for name in train_batch.example_datasets)
    assert all(name == "b" for name in test_batch.example_datasets)


def test_sequence_autoencoder_forward_shape() -> None:
    batch = build_task_context_batch(_task_context_frame())
    encoder = TaskContextTransformerEncoder(input_dim=batch.sequences.shape[-1], hidden_dim=32, num_heads=4, num_layers=1)
    model = TaskContextSequenceAutoencoder(encoder=encoder, hidden_dim=32, seq_len=batch.sequences.shape[1], input_dim=batch.sequences.shape[2])
    recon, embedding = model(batch.sequences)
    assert recon.shape == batch.sequences.shape
    assert embedding.shape == (4, 32)
    assert torch.isfinite(recon).all()


def test_task_context_normalization_preserves_binary_features() -> None:
    batch = build_task_context_batch(_task_context_frame())
    norm = fit_task_context_normalization(batch)
    normalized = apply_task_context_normalization(batch, norm)
    is_anchor_idx = batch.feature_names.index("is_anchor")
    assert torch.equal(batch.sequences[..., is_anchor_idx], normalized.sequences[..., is_anchor_idx])


def test_slice_batch_by_datasets() -> None:
    batch = build_task_context_batch(_task_context_frame())
    sliced = slice_batch_by_datasets(batch, ["b"])
    assert sliced.dataset_names == ("b",)
    assert sliced.example_datasets == ("b", "b")


def test_task_prototype_bank_save_load_and_query(tmp_path) -> None:
    batch = build_task_context_batch(_task_context_frame())
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=torch.float32,
    )
    bank = build_task_prototype_bank(
        embeddings=embeddings,
        batch=batch,
        encoder_kind="transformer",
        hidden_dim=2,
        normalize_features=False,
        normalization=None,
    )
    path = tmp_path / "bank.json"
    save_task_prototype_bank(bank, path)
    loaded = load_task_prototype_bank(path)
    assert loaded.dataset_names == ("a", "b")
    assert loaded.encoder_kind == "transformer"
    result = query_task_prototype_bank(loaded, embeddings[:2], query_dataset="a", top_k=2)
    assert result["known_dataset"] is True
    assert result["exact_reuse_available"] is True
    assert result["top_neighbors"][0]["dataset"] == "a"
    assert result["similar_neighbors_excluding_exact"][0]["dataset"] == "b"


def test_supervised_contrastive_loss_prefers_grouped_embeddings() -> None:
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    grouped = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=torch.float32,
    )
    mixed = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
            [0.1, 0.9],
        ],
        dtype=torch.float32,
    )
    grouped_loss = supervised_contrastive_loss(grouped, labels, temperature=0.1)
    mixed_loss = supervised_contrastive_loss(mixed, labels, temperature=0.1)
    assert grouped_loss < mixed_loss


def test_metadata_neighbor_targets_form_probability_rows() -> None:
    batch = build_task_context_batch(_task_context_frame())
    targets = metadata_neighbor_targets(batch, temperature=0.2)
    assert targets.shape == (2, 2)
    assert torch.allclose(targets.sum(dim=-1), torch.ones(2))
    assert torch.allclose(torch.diag(targets), torch.zeros(2))


def test_neighborhood_consistency_loss_is_finite() -> None:
    batch = build_task_context_batch(_task_context_frame())
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=torch.float32,
    )
    targets = metadata_neighbor_targets(batch, temperature=0.2)
    loss = neighborhood_consistency_loss(embeddings, batch, target_distributions=targets, temperature=0.1)
    pred = embedding_neighbor_distribution(embeddings, batch, temperature=0.1)
    assert torch.isfinite(loss)
    assert pred.shape == targets.shape
