from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


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


@dataclass(frozen=True)
class TaskPrototypeBank:
    dataset_names: tuple[str, ...]
    centroids: dict[str, torch.Tensor]
    counts: dict[str, int]
    feature_names: tuple[str, ...]
    encoder_kind: str
    hidden_dim: int
    normalize_features: bool
    training_objective: str = "reconstruction"
    normalization: TaskContextNormalization | None = None


def _dataset_indices(batch: TaskContextBatch) -> dict[str, list[int]]:
    return {
        dataset: [idx for idx, name in enumerate(batch.example_datasets) if name == dataset]
        for dataset in batch.dataset_names
    }


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


def slice_batch_by_datasets(batch: TaskContextBatch, dataset_names: Sequence[str]) -> TaskContextBatch:
    wanted = set(dataset_names)
    indices = [idx for idx, dataset in enumerate(batch.example_datasets) if dataset in wanted]
    if not indices:
        raise ValueError(f"No examples matched dataset_names={sorted(wanted)}")
    return TaskContextBatch(
        sequences=batch.sequences[indices],
        labels=batch.labels[indices],
        dataset_names=tuple(name for name in batch.dataset_names if name in wanted),
        example_datasets=tuple(batch.example_datasets[idx] for idx in indices),
        feature_names=batch.feature_names,
    )


def normalized_centroids(
    embeddings: torch.Tensor,
    dataset_names: Sequence[str],
    example_datasets: Sequence[str],
) -> dict[str, torch.Tensor]:
    centroids: dict[str, torch.Tensor] = {}
    for dataset in dataset_names:
        idx = [i for i, name in enumerate(example_datasets) if name == dataset]
        if not idx:
            continue
        centroid = embeddings[idx].mean(dim=0)
        centroids[str(dataset)] = F.normalize(centroid.unsqueeze(0), dim=-1).squeeze(0).cpu()
    return centroids


def dataset_signatures_from_sequences(batch: TaskContextBatch) -> torch.Tensor:
    signatures = []
    indices_by_dataset = _dataset_indices(batch)
    for dataset in batch.dataset_names:
        idx = indices_by_dataset[dataset]
        dataset_seq = batch.sequences[idx]
        signatures.append(dataset_seq.mean(dim=0).reshape(-1))
    return torch.stack(signatures, dim=0)


def metadata_neighbor_targets(batch: TaskContextBatch, temperature: float = 0.2) -> torch.Tensor:
    signatures = F.normalize(dataset_signatures_from_sequences(batch), dim=-1)
    sims = signatures @ signatures.T
    mask = torch.eye(sims.shape[0], device=sims.device, dtype=torch.bool)
    logits = sims / temperature
    logits = logits.masked_fill(mask, float("-inf"))
    targets = torch.softmax(logits, dim=-1)
    targets = torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
    return targets


def embedding_neighbor_distribution(embeddings: torch.Tensor, batch: TaskContextBatch, temperature: float = 0.1) -> torch.Tensor:
    centroids = []
    indices_by_dataset = _dataset_indices(batch)
    for dataset in batch.dataset_names:
        idx = indices_by_dataset[dataset]
        centroid = embeddings[idx].mean(dim=0)
        centroids.append(F.normalize(centroid.unsqueeze(0), dim=-1).squeeze(0))
    centroid_matrix = torch.stack(centroids, dim=0)
    sims = centroid_matrix @ centroid_matrix.T
    mask = torch.eye(sims.shape[0], device=sims.device, dtype=torch.bool)
    logits = (sims / temperature).masked_fill(mask, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    return probs


def neighborhood_consistency_loss(
    embeddings: torch.Tensor,
    batch: TaskContextBatch,
    *,
    target_distributions: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    predicted = embedding_neighbor_distribution(embeddings, batch=batch, temperature=temperature)
    target = target_distributions.to(predicted.device)
    valid = target.sum(dim=-1) > 0
    if not torch.any(valid):
        raise ValueError("target_distributions must contain at least one non-empty row")
    loss = target[valid] * (torch.log(target[valid].clamp_min(1e-12)) - torch.log(predicted[valid].clamp_min(1e-12)))
    return loss.sum(dim=-1).mean()


def build_task_prototype_bank(
    embeddings: torch.Tensor,
    batch: TaskContextBatch,
    *,
    encoder_kind: str,
    hidden_dim: int,
    normalize_features: bool,
    training_objective: str = "reconstruction",
    normalization: TaskContextNormalization | None = None,
) -> TaskPrototypeBank:
    centroids = normalized_centroids(embeddings=embeddings, dataset_names=batch.dataset_names, example_datasets=batch.example_datasets)
    counts = {
        dataset: sum(1 for name in batch.example_datasets if name == dataset)
        for dataset in batch.dataset_names
    }
    return TaskPrototypeBank(
        dataset_names=tuple(batch.dataset_names),
        centroids=centroids,
        counts=counts,
        feature_names=tuple(batch.feature_names),
        encoder_kind=encoder_kind,
        hidden_dim=hidden_dim,
        normalize_features=normalize_features,
        training_objective=training_objective,
        normalization=normalization,
    )


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape={tuple(embeddings.shape)}")
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels must be 1D with same batch size as embeddings")
    z = F.normalize(embeddings, dim=-1)
    logits = (z @ z.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    non_self = ~torch.eye(labels.shape[0], device=labels.device, dtype=torch.bool)
    positive_mask = same_label & non_self
    exp_logits = torch.exp(logits) * non_self
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_counts = positive_mask.sum(dim=1)
    valid = positive_counts > 0
    if not torch.any(valid):
        raise ValueError("supervised_contrastive_loss requires at least one positive pair")
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_counts.clamp_min(1)
    return -mean_log_prob_pos[valid].mean()


def _normalization_to_jsonable(normalization: TaskContextNormalization | None) -> dict[str, Any] | None:
    if normalization is None:
        return None
    return {
        "mean": normalization.mean.tolist(),
        "std": normalization.std.tolist(),
        "continuous_mask": normalization.continuous_mask.tolist(),
    }


def _normalization_from_jsonable(payload: dict[str, Any] | None) -> TaskContextNormalization | None:
    if payload is None:
        return None
    return TaskContextNormalization(
        mean=torch.tensor(payload["mean"], dtype=torch.float32),
        std=torch.tensor(payload["std"], dtype=torch.float32),
        continuous_mask=torch.tensor(payload["continuous_mask"], dtype=torch.bool),
    )


def save_task_prototype_bank(bank: TaskPrototypeBank, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_names": list(bank.dataset_names),
        "centroids": {name: tensor.tolist() for name, tensor in bank.centroids.items()},
        "counts": bank.counts,
        "feature_names": list(bank.feature_names),
        "encoder_kind": bank.encoder_kind,
        "hidden_dim": bank.hidden_dim,
        "normalize_features": bank.normalize_features,
        "training_objective": bank.training_objective,
        "normalization": _normalization_to_jsonable(bank.normalization),
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_task_prototype_bank(path: str | Path) -> TaskPrototypeBank:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return TaskPrototypeBank(
        dataset_names=tuple(payload["dataset_names"]),
        centroids={name: torch.tensor(values, dtype=torch.float32) for name, values in payload["centroids"].items()},
        counts={str(name): int(count) for name, count in payload["counts"].items()},
        feature_names=tuple(payload["feature_names"]),
        encoder_kind=str(payload["encoder_kind"]),
        hidden_dim=int(payload["hidden_dim"]),
        normalize_features=bool(payload["normalize_features"]),
        training_objective=str(payload.get("training_objective", "reconstruction")),
        normalization=_normalization_from_jsonable(payload.get("normalization")),
    )


def query_task_prototype_bank(
    bank: TaskPrototypeBank,
    query_embeddings: torch.Tensor,
    *,
    query_dataset: str,
    top_k: int = 3,
) -> dict[str, Any]:
    if query_embeddings.ndim != 2:
        raise ValueError(f"query_embeddings must be 2D, got shape={tuple(query_embeddings.shape)}")
    if not bank.centroids:
        raise ValueError("TaskPrototypeBank has no centroids")
    query_embeddings = F.normalize(query_embeddings, dim=-1)
    centroid_names = sorted(bank.centroids.keys())
    centroid_matrix = torch.stack([bank.centroids[name] for name in centroid_names], dim=0)
    similarities = query_embeddings @ centroid_matrix.T
    probs = torch.softmax(similarities, dim=-1).mean(dim=0)
    probs = probs / probs.sum()
    top_values, top_indices = torch.topk(probs, k=min(top_k, probs.shape[0]))
    top_neighbors = [
        {
            "dataset": centroid_names[idx],
            "probability": float(prob),
            "mean_similarity": float(similarities[:, idx].mean().item()),
        }
        for prob, idx in zip(top_values.tolist(), top_indices.tolist())
    ]
    similar_neighbors = [item for item in top_neighbors if item["dataset"] != query_dataset]
    exact_similarity = None
    if query_dataset in bank.centroids:
        exact_idx = centroid_names.index(query_dataset)
        exact_similarity = float(similarities[:, exact_idx].mean().item())
    return {
        "query_dataset": query_dataset,
        "known_dataset": query_dataset in bank.centroids,
        "exact_reuse_available": query_dataset in bank.centroids,
        "exact_match_mean_similarity": exact_similarity,
        "top_neighbors": top_neighbors,
        "similar_neighbors_excluding_exact": similar_neighbors,
        "soft_neighbor_entropy": float(-(probs * torch.log(probs + 1e-12)).sum().item()),
    }


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
