#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from graphdrone_fit.task_conditioned_prior import (
    TaskContextDatasetClassifier,
    TaskContextGRUEncoder,
    TaskContextNormalization,
    TaskContextSequenceAutoencoder,
    TaskContextTransformerEncoder,
    apply_task_context_normalization,
    build_task_context_batch,
    fit_task_context_normalization,
    split_batch_by_dataset,
)

import pandas as pd


def _build_model(encoder_kind: str, input_dim: int, hidden_dim: int, num_classes: int) -> TaskContextDatasetClassifier:
    if encoder_kind == "transformer":
        encoder = TaskContextTransformerEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
    elif encoder_kind == "gru":
        encoder = TaskContextGRUEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unsupported encoder_kind={encoder_kind!r}")
    return TaskContextDatasetClassifier(encoder=encoder, hidden_dim=hidden_dim, num_classes=num_classes)


def _build_encoder(encoder_kind: str, input_dim: int, hidden_dim: int):
    if encoder_kind == "transformer":
        return TaskContextTransformerEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
    if encoder_kind == "gru":
        return TaskContextGRUEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
    raise ValueError(f"Unsupported encoder_kind={encoder_kind!r}")


def _run_one(encoder_kind: str, batch, epochs: int, lr: float, weight_decay: float, device: str) -> dict:
    model = _build_model(encoder_kind, input_dim=batch.sequences.shape[-1], hidden_dim=64, num_classes=len(batch.dataset_names)).to(device)
    sequences = batch.sequences.to(device)
    labels = batch.labels.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, embedding = model(sequences)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = float((preds == labels).float().mean().item())
            mean_norm = float(embedding.norm(dim=-1).mean().item())
        history.append({"epoch": epoch + 1, "loss": float(loss.item()), "accuracy": acc, "mean_embedding_norm": mean_norm})

    return {
        "encoder_kind": encoder_kind,
        "final_loss": history[-1]["loss"],
        "final_accuracy": history[-1]["accuracy"],
        "history": history,
        "dataset_names": batch.dataset_names,
    }


def _run_leave_one_dataset_out_reconstruction(encoder_kind: str, batch, epochs: int, lr: float, weight_decay: float, device: str) -> dict:
    per_dataset = []
    for held_out_dataset in batch.dataset_names:
        train_batch, test_batch = split_batch_by_dataset(batch, held_out_dataset)
        encoder = _build_encoder(encoder_kind, input_dim=batch.sequences.shape[-1], hidden_dim=64)
        model = TaskContextSequenceAutoencoder(
            encoder=encoder,
            hidden_dim=64,
            seq_len=batch.sequences.shape[1],
            input_dim=batch.sequences.shape[2],
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        train_sequences = train_batch.sequences.to(device)
        test_sequences = test_batch.sequences.to(device)

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            recon, _ = model(train_sequences)
            loss = F.mse_loss(recon, train_sequences)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            train_recon, train_embedding = model(train_sequences)
            test_recon, test_embedding = model(test_sequences)
            train_mse = float(F.mse_loss(train_recon, train_sequences).item())
            test_mse = float(F.mse_loss(test_recon, test_sequences).item())
            per_dataset.append(
                {
                    "held_out_dataset": held_out_dataset,
                    "train_mse": train_mse,
                    "test_mse": test_mse,
                    "generalization_gap": test_mse - train_mse,
                    "train_embedding_norm": float(train_embedding.norm(dim=-1).mean().item()),
                    "test_embedding_norm": float(test_embedding.norm(dim=-1).mean().item()),
                }
            )

    mean_test_mse = sum(item["test_mse"] for item in per_dataset) / len(per_dataset)
    mean_gap = sum(item["generalization_gap"] for item in per_dataset) / len(per_dataset)
    return {
        "encoder_kind": encoder_kind,
        "mean_test_mse": mean_test_mse,
        "mean_generalization_gap": mean_gap,
        "per_dataset": per_dataset,
    }


def _normalized_centroids(embeddings: torch.Tensor, dataset_names: tuple[str, ...], example_datasets: tuple[str, ...]) -> dict[str, torch.Tensor]:
    centroids: dict[str, torch.Tensor] = {}
    for dataset in dataset_names:
        idx = [i for i, name in enumerate(example_datasets) if name == dataset]
        if not idx:
            continue
        centroid = embeddings[idx].mean(dim=0)
        centroids[dataset] = F.normalize(centroid.unsqueeze(0), dim=-1).squeeze(0)
    return centroids


def _entropy_from_counts(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return float("nan")
    probs = [count / total for count in counts.values() if count > 0]
    return float(-sum(p * math.log(p + 1e-12) for p in probs))


def _softmax_neighbor_summary(similarities: torch.Tensor, seen_datasets: list[str], top_k: int = 3) -> dict[str, object]:
    probs = torch.softmax(similarities, dim=-1).mean(dim=0)
    probs = probs / probs.sum()
    values, indices = torch.topk(probs, k=min(top_k, probs.shape[0]))
    top_neighbors = [
        {"dataset": seen_datasets[idx], "probability": float(val)}
        for val, idx in zip(values.tolist(), indices.tolist())
    ]
    entropy = float(-(probs * torch.log(probs + 1e-12)).sum().item())
    return {
        "top_neighbors": top_neighbors,
        "soft_neighbor_entropy": entropy,
    }


def _run_leave_one_dataset_out_similarity(
    encoder_kind: str,
    batch,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
    normalize_features: bool,
) -> dict:
    per_dataset = []
    for held_out_dataset in batch.dataset_names:
        train_batch, test_batch = split_batch_by_dataset(batch, held_out_dataset)
        if normalize_features:
            normalization = fit_task_context_normalization(train_batch)
            train_batch = apply_task_context_normalization(train_batch, normalization)
            test_batch = apply_task_context_normalization(test_batch, normalization)
        encoder = _build_encoder(encoder_kind, input_dim=batch.sequences.shape[-1], hidden_dim=64)
        model = TaskContextSequenceAutoencoder(
            encoder=encoder,
            hidden_dim=64,
            seq_len=batch.sequences.shape[1],
            input_dim=batch.sequences.shape[2],
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        train_sequences = train_batch.sequences.to(device)
        test_sequences = test_batch.sequences.to(device)

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            recon, _ = model(train_sequences)
            loss = F.mse_loss(recon, train_sequences)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            _, train_embedding = model(train_sequences)
            _, test_embedding = model(test_sequences)
            train_embedding = F.normalize(train_embedding, dim=-1)
            test_embedding = F.normalize(test_embedding, dim=-1)
            centroids = _normalized_centroids(train_embedding, train_batch.dataset_names, train_batch.example_datasets)
            seen_datasets = sorted(centroids.keys())
            centroid_matrix = torch.stack([centroids[name] for name in seen_datasets], dim=0)
            sims = test_embedding @ centroid_matrix.T
            top_vals, top_idx = sims.max(dim=1)
            soft_summary = _softmax_neighbor_summary(sims, seen_datasets=seen_datasets, top_k=min(3, len(seen_datasets)))
            votes: dict[str, int] = {}
            for idx in top_idx.tolist():
                name = seen_datasets[idx]
                votes[name] = votes.get(name, 0) + 1
            top_neighbor = max(votes, key=votes.get)
            per_dataset.append(
                {
                    "held_out_dataset": held_out_dataset,
                    "top_neighbor_dataset": top_neighbor,
                    "top_neighbor_fraction": float(votes[top_neighbor] / len(top_idx)),
                    "neighbor_vote_entropy": _entropy_from_counts(votes),
                    "mean_top_similarity": float(top_vals.mean().item()),
                    "min_top_similarity": float(top_vals.min().item()),
                    "max_top_similarity": float(top_vals.max().item()),
                    "neighbor_votes": votes,
                    "soft_neighbor_entropy": soft_summary["soft_neighbor_entropy"],
                    "top_neighbors": soft_summary["top_neighbors"],
                }
            )

    return {
        "encoder_kind": encoder_kind,
        "normalize_features": normalize_features,
        "per_dataset": per_dataset,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype a task-conditioned LMA prior from task-context bootstrap summaries.")
    parser.add_argument("--analysis-dir", type=Path, required=True, help="Directory containing task_context_examples.csv")
    parser.add_argument("--encoder", choices=["transformer", "gru", "both"], default="both")
    parser.add_argument(
        "--mode",
        choices=["dataset_id", "leave_one_dataset_out_reconstruction", "leave_one_dataset_out_similarity"],
        default="dataset_id",
    )
    parser.add_argument("--normalize-features", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    task_context_path = args.analysis_dir / "task_context_examples.csv"
    if not task_context_path.exists():
        raise FileNotFoundError(f"Missing {task_context_path}; rerun extractor with --bootstrap-summaries")
    task_context_df = pd.read_csv(task_context_path)
    batch = build_task_context_batch(task_context_df)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder_kinds = ["transformer", "gru"] if args.encoder == "both" else [args.encoder]
    summary = {
        "analysis_dir": str(args.analysis_dir),
        "mode": args.mode,
        "device": device,
        "normalize_features": bool(args.normalize_features),
        "n_examples": int(batch.sequences.shape[0]),
        "seq_len": int(batch.sequences.shape[1]),
        "input_dim": int(batch.sequences.shape[2]),
        "dataset_names": list(batch.dataset_names),
        "results": [],
    }
    for encoder_kind in encoder_kinds:
        if args.mode == "dataset_id":
            result = _run_one(encoder_kind, batch=batch, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, device=device)
            summary["results"].append({k: v for k, v in result.items() if k != "history"})
            (args.output_dir / f"{encoder_kind}_history.json").write_text(json.dumps(result["history"], indent=2), encoding="utf-8")
        elif args.mode == "leave_one_dataset_out_reconstruction":
            result = _run_leave_one_dataset_out_reconstruction(
                encoder_kind,
                batch=batch,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
            )
            summary["results"].append(result)
        else:
            result = _run_leave_one_dataset_out_similarity(
                encoder_kind,
                batch=batch,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
                normalize_features=args.normalize_features,
            )
            summary["results"].append(result)

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
