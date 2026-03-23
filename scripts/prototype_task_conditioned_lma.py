#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from graphdrone_fit.task_conditioned_prior import (
    TaskContextDatasetClassifier,
    TaskContextGRUEncoder,
    TaskContextTransformerEncoder,
    build_task_context_batch,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype a task-conditioned LMA prior from task-context bootstrap summaries.")
    parser.add_argument("--analysis-dir", type=Path, required=True, help="Directory containing task_context_examples.csv")
    parser.add_argument("--encoder", choices=["transformer", "gru", "both"], default="both")
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
        "device": device,
        "n_examples": int(batch.sequences.shape[0]),
        "seq_len": int(batch.sequences.shape[1]),
        "input_dim": int(batch.sequences.shape[2]),
        "dataset_names": list(batch.dataset_names),
        "results": [],
    }
    for encoder_kind in encoder_kinds:
        result = _run_one(encoder_kind, batch=batch, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, device=device)
        summary["results"].append({k: v for k, v in result.items() if k != "history"})
        (args.output_dir / f"{encoder_kind}_history.json").write_text(json.dumps(result["history"], indent=2), encoding="utf-8")

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
