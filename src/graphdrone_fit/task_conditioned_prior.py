from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Sequence

import pandas as pd
import torch
import torch.nn as nn


CANONICAL_FAMILY_ORDER = ("FULL", "structural_subspace", "local_support", "learned_regime", "domain_semantic", "bootstrap")
CANONICAL_PROJECTION_ORDER = ("identity_subselect", "external_transform", "support_defined", "opaque", "structural_subspace")


def _expert_sort_key(expert_id: str) -> tuple[int, str]:
    if expert_id == "FULL":
        return (0, expert_id)
    if expert_id.startswith("SUB"):
        suffix = expert_id[3:]
        if suffix.isdigit():
            return (1, f"{int(suffix):04d}")
    return (2, expert_id)


@dataclass(frozen=True)
class TaskContextBatch:
    sequences: torch.Tensor
    labels: torch.Tensor
    dataset_names: tuple[str, ...]
    example_datasets: tuple[str, ...]
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class TaskContextNormalization:
    mean: torch.Tensor
    std: torch.Tensor
    continuous_mask: torch.Tensor


def _feature_row(row: pd.Series) -> tuple[list[float], list[str]]:
    mean_token = json.loads(row["mean_token_json"])
    values: list[float] = [float(row["is_anchor"]), float(row["input_dim"]), float(row["preferred_k"]), float(row["mean_norm"]), float(row["std_norm"])]
    names: list[str] = ["is_anchor", "input_dim", "preferred_k", "mean_norm", "std_norm"]
    for family in CANONICAL_FAMILY_ORDER:
        values.append(1.0 if row["family"] == family else 0.0)
        names.append(f"family_{family}")
    for projection in CANONICAL_PROJECTION_ORDER:
        values.append(1.0 if row["projection_kind"] == projection else 0.0)
        names.append(f"projection_{projection}")
    values.extend(float(v) for v in mean_token)
    names.extend(f"mean_token_{idx}" for idx in range(len(mean_token)))
    return values, names


def build_task_context_batch(task_context_df: pd.DataFrame) -> TaskContextBatch:
    required = {"dataset", "bootstrap_id", "expert_id", "family", "projection_kind", "input_dim", "preferred_k", "is_anchor", "mean_norm", "std_norm", "mean_token_json"}
    missing = required - set(task_context_df.columns)
    if missing:
        raise ValueError(f"task_context_df missing required columns: {sorted(missing)}")
    grouped = []
    dataset_names = sorted(task_context_df["dataset"].astype(str).unique().tolist())
    label_by_dataset = {name: idx for idx, name in enumerate(dataset_names)}
    feature_names: tuple[str, ...] | None = None

    for (dataset, bootstrap_id), group in task_context_df.groupby(["dataset", "bootstrap_id"], sort=True):
        ordered = group.sort_values(by="expert_id", key=lambda s: s.map(_expert_sort_key))
        seq_rows = []
        for _, row in ordered.iterrows():
            values, row_feature_names = _feature_row(row)
            if feature_names is None:
                feature_names = tuple(row_feature_names)
            seq_rows.append(values)
        grouped.append((str(dataset), int(bootstrap_id), seq_rows))

    if not grouped:
        raise ValueError("No task-context rows available")
    seq_len = len(grouped[0][2])
    feat_dim = len(grouped[0][2][0])
    for dataset, bootstrap_id, seq_rows in grouped:
        if len(seq_rows) != seq_len:
            raise ValueError(f"Inconsistent sequence length for {(dataset, bootstrap_id)}")
        if any(len(row) != feat_dim for row in seq_rows):
            raise ValueError(f"Inconsistent feature dimension for {(dataset, bootstrap_id)}")

    example_datasets = tuple(dataset for dataset, _, _ in grouped)
    sequences = torch.tensor([seq_rows for _, _, seq_rows in grouped], dtype=torch.float32)
    labels = torch.tensor([label_by_dataset[dataset] for dataset, _, _ in grouped], dtype=torch.long)
    return TaskContextBatch(
        sequences=sequences,
        labels=labels,
        dataset_names=tuple(dataset_names),
        example_datasets=example_datasets,
        feature_names=feature_names or (),
    )


def _continuous_feature_mask(feature_names: Sequence[str]) -> torch.Tensor:
    mask = []
    for name in feature_names:
        is_binary = (
            name == "is_anchor"
            or name.startswith("family_")
            or name.startswith("projection_")
        )
        mask.append(not is_binary)
    return torch.tensor(mask, dtype=torch.bool)


def fit_task_context_normalization(batch: TaskContextBatch) -> TaskContextNormalization:
    mask = _continuous_feature_mask(batch.feature_names)
    flat = batch.sequences.reshape(-1, batch.sequences.shape[-1])
    mean = flat.mean(dim=0)
    std = flat.std(dim=0, unbiased=False).clamp_min(1e-6)
    return TaskContextNormalization(mean=mean, std=std, continuous_mask=mask)


def apply_task_context_normalization(batch: TaskContextBatch, normalization: TaskContextNormalization) -> TaskContextBatch:
    sequences = batch.sequences.clone()
    mask = normalization.continuous_mask
    sequences[..., mask] = (sequences[..., mask] - normalization.mean[mask]) / normalization.std[mask]
    return TaskContextBatch(
        sequences=sequences,
        labels=batch.labels,
        dataset_names=batch.dataset_names,
        example_datasets=batch.example_datasets,
        feature_names=batch.feature_names,
    )


def split_batch_by_dataset(batch: TaskContextBatch, held_out_dataset: str) -> tuple[TaskContextBatch, TaskContextBatch]:
    train_idx = [idx for idx, dataset in enumerate(batch.example_datasets) if dataset != held_out_dataset]
    test_idx = [idx for idx, dataset in enumerate(batch.example_datasets) if dataset == held_out_dataset]
    if not train_idx or not test_idx:
        raise ValueError(f"held_out_dataset={held_out_dataset!r} must have both train and test coverage")

    def _slice(indices: list[int]) -> TaskContextBatch:
        return TaskContextBatch(
            sequences=batch.sequences[indices],
            labels=batch.labels[indices],
            dataset_names=batch.dataset_names,
            example_datasets=tuple(batch.example_datasets[idx] for idx in indices),
            feature_names=batch.feature_names,
        )

    return _slice(train_idx), _slice(test_idx)


class TaskContextTransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, 32, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(sequences)
        batch_size, seq_len, _ = x.shape
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, : seq_len + 1, :]
        encoded = self.encoder(x)
        return self.norm(encoded[:, 0, :])


class TaskContextGRUEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(sequences)
        _, hidden = self.gru(x)
        return self.norm(hidden[-1])


class TaskContextDatasetClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(sequences)
        logits = self.classifier(embedding)
        return logits, embedding


class TaskContextSequenceAutoencoder(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int, seq_len: int, input_dim: int):
        super().__init__()
        self.encoder = encoder
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, seq_len * input_dim),
        )

    def forward(self, sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(sequences)
        recon = self.decoder(embedding).view(sequences.shape[0], self.seq_len, self.input_dim)
        return recon, embedding
