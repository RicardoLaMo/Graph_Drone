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
    TaskContextSequenceAutoencoder,
    TaskContextTransformerEncoder,
    apply_task_context_normalization,
    build_task_prototype_bank,
    build_task_context_batch,
    fit_task_context_normalization,
    load_task_prototype_bank,
    query_task_prototype_bank,
    save_task_prototype_bank,
    slice_batch_by_datasets,
    split_batch_by_dataset,
    supervised_contrastive_loss,
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
            centroids = build_task_prototype_bank(
                embeddings=train_embedding,
                batch=train_batch,
                encoder_kind=encoder_kind,
                hidden_dim=64,
                normalize_features=normalize_features,
                normalization=normalization if normalize_features else None,
            ).centroids
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


def _fit_similarity_encoder(
    encoder_kind: str,
    batch,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
):
    encoder = _build_encoder(encoder_kind, input_dim=batch.sequences.shape[-1], hidden_dim=64)
    model = TaskContextSequenceAutoencoder(
        encoder=encoder,
        hidden_dim=64,
        seq_len=batch.sequences.shape[1],
        input_dim=batch.sequences.shape[2],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sequences = batch.sequences.to(device)
    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        recon, embedding = model(sequences)
        loss = F.mse_loss(recon, sequences)
        loss.backward()
        optimizer.step()
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(loss.item()),
                "mean_embedding_norm": float(embedding.norm(dim=-1).mean().item()),
            }
        )
    return model, history


def _fit_contrastive_similarity_encoder(
    encoder_kind: str,
    batch,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
    contrastive_temperature: float,
    reconstruction_weight: float,
):
    encoder = _build_encoder(encoder_kind, input_dim=batch.sequences.shape[-1], hidden_dim=64)
    model = TaskContextSequenceAutoencoder(
        encoder=encoder,
        hidden_dim=64,
        seq_len=batch.sequences.shape[1],
        input_dim=batch.sequences.shape[2],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sequences = batch.sequences.to(device)
    labels = batch.labels.to(device)
    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        recon, embedding = model(sequences)
        recon_loss = F.mse_loss(recon, sequences)
        contrastive_loss = supervised_contrastive_loss(embedding, labels, temperature=contrastive_temperature)
        loss = contrastive_loss + reconstruction_weight * recon_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            z = F.normalize(embedding, dim=-1)
            sims = z @ z.T
            same = labels.unsqueeze(0) == labels.unsqueeze(1)
            non_self = ~torch.eye(labels.shape[0], device=labels.device, dtype=torch.bool)
            pos = sims[same & non_self]
            neg = sims[(~same) & non_self]
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(loss.item()),
                    "contrastive_loss": float(contrastive_loss.item()),
                    "reconstruction_loss": float(recon_loss.item()),
                    "positive_similarity": float(pos.mean().item()),
                    "negative_similarity": float(neg.mean().item()),
                }
            )
    return model, history


def _run_fit_prototype_bank(
    encoder_kind: str,
    batch,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
    normalize_features: bool,
    output_dir: Path,
) -> dict:
    normalization = None
    working_batch = batch
    if normalize_features:
        normalization = fit_task_context_normalization(batch)
        working_batch = apply_task_context_normalization(batch, normalization)
    model, history = _fit_similarity_encoder(
        encoder_kind=encoder_kind,
        batch=working_batch,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )
    with torch.no_grad():
        model.eval()
        _, embedding = model(working_batch.sequences.to(device))
        embedding = F.normalize(embedding, dim=-1).cpu()
    bank = build_task_prototype_bank(
        embeddings=embedding,
        batch=working_batch,
        encoder_kind=encoder_kind,
        hidden_dim=64,
        normalize_features=normalize_features,
        normalization=normalization,
    )
    bank_path = output_dir / f"{encoder_kind}_prototype_bank.json"
    checkpoint_path = output_dir / f"{encoder_kind}_encoder_state.pt"
    save_task_prototype_bank(bank, bank_path)
    torch.save(
        {
            "encoder_kind": encoder_kind,
            "hidden_dim": 64,
            "state_dict": model.encoder.state_dict(),
        },
        checkpoint_path,
    )
    return {
        "encoder_kind": encoder_kind,
        "normalize_features": normalize_features,
        "prototype_bank_path": str(bank_path),
        "encoder_checkpoint_path": str(checkpoint_path),
        "n_datasets": len(bank.dataset_names),
        "dataset_names": list(bank.dataset_names),
        "counts": bank.counts,
        "history_tail": history[-5:],
    }


def _run_fit_contrastive_prototype_bank(
    encoder_kind: str,
    batch,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
    normalize_features: bool,
    output_dir: Path,
    contrastive_temperature: float,
    reconstruction_weight: float,
) -> dict:
    normalization = None
    working_batch = batch
    if normalize_features:
        normalization = fit_task_context_normalization(batch)
        working_batch = apply_task_context_normalization(batch, normalization)
    model, history = _fit_contrastive_similarity_encoder(
        encoder_kind=encoder_kind,
        batch=working_batch,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        contrastive_temperature=contrastive_temperature,
        reconstruction_weight=reconstruction_weight,
    )
    with torch.no_grad():
        model.eval()
        _, embedding = model(working_batch.sequences.to(device))
        embedding = F.normalize(embedding, dim=-1).cpu()
    bank = build_task_prototype_bank(
        embeddings=embedding,
        batch=working_batch,
        encoder_kind=encoder_kind,
        hidden_dim=64,
        normalize_features=normalize_features,
        training_objective="contrastive_reconstruction",
        normalization=normalization,
    )
    bank_path = output_dir / f"{encoder_kind}_prototype_bank.json"
    checkpoint_path = output_dir / f"{encoder_kind}_encoder_state.pt"
    save_task_prototype_bank(bank, bank_path)
    torch.save(
        {
            "encoder_kind": encoder_kind,
            "hidden_dim": 64,
            "state_dict": model.encoder.state_dict(),
            "training_objective": "contrastive_reconstruction",
            "contrastive_temperature": contrastive_temperature,
            "reconstruction_weight": reconstruction_weight,
        },
        checkpoint_path,
    )
    return {
        "encoder_kind": encoder_kind,
        "normalize_features": normalize_features,
        "prototype_bank_path": str(bank_path),
        "encoder_checkpoint_path": str(checkpoint_path),
        "training_objective": "contrastive_reconstruction",
        "contrastive_temperature": contrastive_temperature,
        "reconstruction_weight": reconstruction_weight,
        "n_datasets": len(bank.dataset_names),
        "dataset_names": list(bank.dataset_names),
        "counts": bank.counts,
        "history_tail": history[-5:],
    }


def _run_query_prototype_bank(
    encoder_kind: str,
    batch,
    device: str,
    bank_dir: Path,
    query_datasets: list[str] | None,
) -> dict:
    bank_path = bank_dir / f"{encoder_kind}_prototype_bank.json"
    checkpoint_path = bank_dir / f"{encoder_kind}_encoder_state.pt"
    bank = load_task_prototype_bank(bank_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    encoder = _build_encoder(
        encoder_kind=checkpoint["encoder_kind"],
        input_dim=batch.sequences.shape[-1],
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)
    encoder.load_state_dict(checkpoint["state_dict"])
    encoder.eval()

    working_batch = batch
    if query_datasets:
        working_batch = slice_batch_by_datasets(working_batch, query_datasets)
    if bank.normalize_features and bank.normalization is not None:
        working_batch = apply_task_context_normalization(working_batch, bank.normalization)

    results = []
    for dataset in working_batch.dataset_names:
        dataset_batch = slice_batch_by_datasets(working_batch, [dataset])
        with torch.no_grad():
            embedding = encoder(dataset_batch.sequences.to(device))
            embedding = F.normalize(embedding, dim=-1).cpu()
        result = query_task_prototype_bank(bank, embedding, query_dataset=dataset, top_k=min(3, len(bank.dataset_names)))
        results.append(result)

    return {
        "encoder_kind": encoder_kind,
        "prototype_bank_path": str(bank_path),
        "training_objective": bank.training_objective,
        "query_datasets": list(working_batch.dataset_names),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype a task-conditioned LMA prior from task-context bootstrap summaries.")
    parser.add_argument("--analysis-dir", type=Path, required=True, help="Directory containing task_context_examples.csv")
    parser.add_argument("--encoder", choices=["transformer", "gru", "both"], default="both")
    parser.add_argument(
        "--mode",
        choices=[
            "dataset_id",
            "leave_one_dataset_out_reconstruction",
            "leave_one_dataset_out_similarity",
            "fit_prototype_bank",
            "fit_contrastive_prototype_bank",
            "query_prototype_bank",
        ],
        default="dataset_id",
    )
    parser.add_argument("--normalize-features", action="store_true")
    parser.add_argument("--bank-dir", type=Path, help="Directory containing saved prototype bank artifacts for query mode.")
    parser.add_argument("--query-datasets", nargs="+", help="Optional dataset names to query from the provided analysis dir.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--reconstruction-weight", type=float, default=0.25)
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
        elif args.mode == "leave_one_dataset_out_similarity":
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
        elif args.mode == "fit_prototype_bank":
            result = _run_fit_prototype_bank(
                encoder_kind,
                batch=batch,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
                normalize_features=args.normalize_features,
                output_dir=args.output_dir,
            )
            summary["results"].append(result)
        elif args.mode == "fit_contrastive_prototype_bank":
            result = _run_fit_contrastive_prototype_bank(
                encoder_kind,
                batch=batch,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
                normalize_features=args.normalize_features,
                output_dir=args.output_dir,
                contrastive_temperature=args.contrastive_temperature,
                reconstruction_weight=args.reconstruction_weight,
            )
            summary["results"].append(result)
        else:
            if args.bank_dir is None:
                raise ValueError("--bank-dir is required for query_prototype_bank mode")
            result = _run_query_prototype_bank(
                encoder_kind,
                batch=batch,
                device=device,
                bank_dir=args.bank_dir,
                query_datasets=args.query_datasets,
            )
            summary["results"].append(result)

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
