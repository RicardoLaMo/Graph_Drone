#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from graphdrone_fit.task_conditioned_prior import (
    apply_task_context_normalization,
    build_task_context_batch,
    build_task_context_encoder,
    load_task_prototype_bank,
    save_task_prototype_bank,
    slice_batch_by_datasets,
    update_task_prototype_bank_feedback,
)


def _classification_reward(row: pd.Series) -> float:
    reward = 0.0
    reward += float(row.get("f1_delta", 0.0))
    reward += float(row.get("log_loss_rel_improvement", 0.0))
    reward += 0.25 * float(row.get("auc_roc_delta", 0.0))
    reward += 0.25 * float(row.get("pr_auc_delta", 0.0))
    reward -= 0.10 * max(float(row.get("defer_delta", 0.0)), 0.0)
    return reward


def _regression_reward(row: pd.Series) -> float:
    reward = 0.0
    reward += float(row.get("rmse_rel_improvement", 0.0))
    reward += 0.25 * float(row.get("r2_delta", 0.0))
    reward -= 0.10 * max(-float(row.get("latency_improvement", 0.0)), 0.0)
    return reward


def _row_reward(row: pd.Series) -> float:
    task_type = str(row.get("task_type", "classification"))
    if task_type == "regression":
        return _regression_reward(row)
    return _classification_reward(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update a task prototype bank from champion/challenger outcome feedback.")
    parser.add_argument("--analysis-dir", type=Path, required=True, help="Directory containing task_context_examples.csv")
    parser.add_argument("--bank-dir", type=Path, required=True, help="Directory containing prototype bank and encoder checkpoint")
    parser.add_argument("--comparison-csv", type=Path, required=True, help="paired_task_deltas.csv from champion/challenger")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination for updated bank artifacts")
    parser.add_argument("--encoder-kind", choices=["transformer", "gru"], default="transformer")
    parser.add_argument("--feedback-blend", type=float, default=0.75)
    parser.add_argument("--feedback-temperature", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    task_context_path = args.analysis_dir / "task_context_examples.csv"
    if not task_context_path.exists():
        raise FileNotFoundError(f"Missing {task_context_path}")
    task_context_df = pd.read_csv(task_context_path)
    full_batch = build_task_context_batch(task_context_df)

    bank_path = args.bank_dir / f"{args.encoder_kind}_prototype_bank.json"
    ckpt_path = args.bank_dir / f"{args.encoder_kind}_encoder_state.pt"
    bank = load_task_prototype_bank(bank_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    encoder = build_task_context_encoder(
        encoder_kind=str(checkpoint["encoder_kind"]),
        input_dim=full_batch.sequences.shape[-1],
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(args.device)
    encoder.load_state_dict(checkpoint["state_dict"])
    encoder.eval()

    comparison = pd.read_csv(args.comparison_csv)
    updates: list[dict[str, object]] = []
    working_bank = bank
    for _, row in comparison.iterrows():
        dataset = str(row["dataset"])
        dataset_batch = slice_batch_by_datasets(full_batch, [dataset])
        if working_bank.normalize_features and working_bank.normalization is not None:
            dataset_batch = apply_task_context_normalization(dataset_batch, working_bank.normalization)
        with torch.no_grad():
            embedding = encoder(dataset_batch.sequences.to(args.device))
            embedding = torch.nn.functional.normalize(embedding, dim=-1).cpu()

        reward = _row_reward(row)
        top_neighbor = str(row.get("task_prior_top_neighbor", "") or "")
        top_neighbor_prob = float(row.get("task_prior_top_neighbor_prob", 0.0) or 0.0)
        neighbor_rewards = {}
        if top_neighbor:
            neighbor_rewards[top_neighbor] = reward * max(top_neighbor_prob, 1e-6)

        working_bank = update_task_prototype_bank_feedback(
            working_bank,
            query_dataset=dataset,
            query_embeddings=embedding,
            reward=reward,
            neighbor_rewards=neighbor_rewards,
        )
        updates.append(
            {
                "dataset": dataset,
                "reward": reward,
                "top_neighbor": top_neighbor,
                "top_neighbor_prob": top_neighbor_prob,
                "neighbor_reward": neighbor_rewards.get(top_neighbor, 0.0),
            }
        )

    working_bank = type(working_bank)(
        dataset_names=working_bank.dataset_names,
        centroids=working_bank.centroids,
        counts=working_bank.counts,
        feature_names=working_bank.feature_names,
        encoder_kind=working_bank.encoder_kind,
        hidden_dim=working_bank.hidden_dim,
        normalize_features=working_bank.normalize_features,
        training_objective=working_bank.training_objective,
        normalization=working_bank.normalization,
        feedback_blend=args.feedback_blend,
        feedback_temperature=args.feedback_temperature,
        dataset_feedback=working_bank.dataset_feedback,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_task_prototype_bank(working_bank, args.output_dir / f"{args.encoder_kind}_prototype_bank.json")
    shutil.copy2(ckpt_path, args.output_dir / f"{args.encoder_kind}_encoder_state.pt")
    summary = {
        "analysis_dir": str(args.analysis_dir),
        "source_bank_dir": str(args.bank_dir),
        "comparison_csv": str(args.comparison_csv),
        "encoder_kind": args.encoder_kind,
        "feedback_blend": args.feedback_blend,
        "feedback_temperature": args.feedback_temperature,
        "n_updates": len(updates),
        "updates": updates,
    }
    (args.output_dir / "feedback_update_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
