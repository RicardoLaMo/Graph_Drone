#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from graphdrone_fit import (
    ExpertBuildSpec,
    GraphDrone,
    IdentitySelectorAdapter,
    ViewDescriptor,
)
from graphdrone_fit.presets import build_graphdrone_config_from_preset


def _load_benchmark_module():
    import importlib.util

    module_path = ROOT / "scripts" / "run_geopoe_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_geopoe_benchmark_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import benchmark module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark_module()
ALL_DATASETS = BENCHMARK.ALL_DATASETS
load_dataset = BENCHMARK.load_dataset


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_specs(n_features: int, *, task_type: str, seed: int = 42) -> tuple[ExpertBuildSpec, ...]:
    full_idx = tuple(range(n_features))
    dev = _device()
    params = {"n_estimators": 8, "device": dev}

    full_kind = "foundation_regressor" if task_type == "regression" else "foundation_classifier"
    full_spec = ExpertBuildSpec(
        descriptor=ViewDescriptor(
            expert_id="FULL",
            family="FULL",
            view_name="Foundation Full",
            is_anchor=True,
            input_dim=n_features,
            input_indices=full_idx,
        ),
        model_kind=full_kind,
        input_adapter=IdentitySelectorAdapter(indices=full_idx),
        model_params=params,
    )

    sub_specs: list[ExpertBuildSpec] = []
    for sub_seed, sub_frac in [(0, 0.7), (1, 0.7), (2, 0.8)]:
        rng = np.random.RandomState(seed + sub_seed)
        size = max(1, int(n_features * sub_frac))
        idx = tuple(sorted(rng.choice(n_features, size, replace=False).tolist()))
        sub_specs.append(
            ExpertBuildSpec(
                descriptor=ViewDescriptor(
                    expert_id=f"SUB{sub_seed}",
                    family="structural_subspace",
                    view_name=f"Foundation Sub {sub_seed}",
                    input_dim=size,
                    input_indices=idx,
                ),
                model_kind=full_kind,
                input_adapter=IdentitySelectorAdapter(indices=idx),
                model_params=params,
            )
        )
    return (full_spec, *sub_specs)


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm <= 1e-12 or b_norm <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (a_norm * b_norm))


def _grouped_similarity_summary(similarity_df: pd.DataFrame) -> pd.DataFrame:
    if similarity_df.empty:
        return pd.DataFrame(
            columns=[
                "same_task_type",
                "same_family",
                "same_anchor_flag",
                "same_dataset",
                "pair_count",
                "mean_cosine_similarity",
                "median_cosine_similarity",
            ]
        )
    grouped = (
        similarity_df.groupby(
            ["same_task_type", "same_family", "same_anchor_flag", "same_dataset"],
            dropna=False,
        )["cosine_similarity"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .rename(
            columns={
                "count": "pair_count",
                "mean": "mean_cosine_similarity",
                "median": "median_cosine_similarity",
            }
        )
        .sort_values(
            by=["same_task_type", "same_family", "same_anchor_flag", "same_dataset"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )
    return grouped


def _dataset_rows(datasets: Iterable[str], *, fold: int, max_samples: int, sample_rows: int, preset: str) -> tuple[list[dict], list[dict], list[dict]]:
    descriptor_rows: list[dict] = []
    token_rows: list[dict] = []
    similarity_rows: list[dict] = []
    mean_vectors: dict[tuple[str, str], np.ndarray] = {}
    meta_by_key: dict[tuple[str, str], dict] = {}

    for dataset in datasets:
        X, y, task_type = load_dataset(dataset, max_samples=max_samples)
        X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=fold * 17 + 42)
        n_classes = int(len(np.unique(y))) if task_type == "classification" else 1
        cfg = build_graphdrone_config_from_preset(
            preset=preset,
            n_classes=n_classes,
            default_router_kind="contextual_transformer" if task_type == "regression" else "bootstrap_full_only",
        )
        model = GraphDrone(cfg)
        specs = _build_specs(X_tr.shape[1], task_type=task_type)
        model.fit(X_tr, y_tr, expert_specs=specs, problem_type="regression" if task_type == "regression" else None)

        sample_n = min(len(X_te), sample_rows)
        sample_idx = np.random.RandomState(42).choice(len(X_te), size=sample_n, replace=False)
        X_sample = X_te[sample_idx]
        batch = model._expert_factory.predict_all(X_sample)
        token_batch = (
            model._build_regression_tokens(X_sample, batch)
            if task_type == "regression"
            else model._build_classification_tokens(X_sample, batch)
        )
        tokens = token_batch.tokens.detach().cpu().numpy()

        for expert_idx, descriptor in enumerate(batch.descriptors):
            descriptor_rows.append(
                {
                    "dataset": dataset,
                    "task_type": task_type,
                    "expert_id": descriptor.expert_id,
                    "family": descriptor.family,
                    "view_name": descriptor.view_name,
                    "projection_kind": descriptor.projection_kind,
                    "input_dim": descriptor.input_dim,
                    "preferred_k": descriptor.preferred_k,
                    "is_anchor": int(descriptor.is_anchor),
                    "n_token_rows": sample_n,
                    "token_dim": int(tokens.shape[-1]),
                }
            )

            expert_tokens = tokens[:, expert_idx, :]
            mean_vec = expert_tokens.mean(axis=0)
            std_vec = expert_tokens.std(axis=0)
            key = (dataset, descriptor.expert_id)
            mean_vectors[key] = mean_vec
            meta_by_key[key] = {
                "dataset": dataset,
                "task_type": task_type,
                "expert_id": descriptor.expert_id,
                "family": descriptor.family,
                "is_anchor": int(descriptor.is_anchor),
            }
            token_rows.append(
                {
                    "dataset": dataset,
                    "task_type": task_type,
                    "expert_id": descriptor.expert_id,
                    "family": descriptor.family,
                    "is_anchor": int(descriptor.is_anchor),
                    "token_dim": int(tokens.shape[-1]),
                    "mean_norm": float(np.linalg.norm(mean_vec)),
                    "std_norm": float(np.linalg.norm(std_vec)),
                    "mean_token_json": json.dumps(mean_vec.tolist()),
                }
            )

    keys = list(mean_vectors)
    for idx, key_a in enumerate(keys):
        for key_b in keys[idx + 1 :]:
            meta_a = meta_by_key[key_a]
            meta_b = meta_by_key[key_b]
            similarity_rows.append(
                {
                    "dataset_a": meta_a["dataset"],
                    "task_type_a": meta_a["task_type"],
                    "expert_id_a": meta_a["expert_id"],
                    "family_a": meta_a["family"],
                    "is_anchor_a": meta_a["is_anchor"],
                    "dataset_b": meta_b["dataset"],
                    "task_type_b": meta_b["task_type"],
                    "expert_id_b": meta_b["expert_id"],
                    "family_b": meta_b["family"],
                    "is_anchor_b": meta_b["is_anchor"],
                    "same_dataset": int(meta_a["dataset"] == meta_b["dataset"]),
                    "same_task_type": int(meta_a["task_type"] == meta_b["task_type"]),
                    "same_family": int(meta_a["family"] == meta_b["family"]),
                    "same_anchor_flag": int(meta_a["is_anchor"] == meta_b["is_anchor"]),
                    "cosine_similarity": _safe_cosine(mean_vectors[key_a], mean_vectors[key_b]),
                }
            )

    return descriptor_rows, token_rows, similarity_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline cross-dataset view-token analysis for AFC LMA research.")
    parser.add_argument("--datasets", nargs="+", required=True, choices=sorted(ALL_DATASETS.keys()))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--sample-rows", type=int, default=128)
    parser.add_argument("--preset", default="v1_20_champion")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "eval" / "afc_cross_dataset_lma")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    descriptor_rows, token_rows, similarity_rows = _dataset_rows(
        args.datasets,
        fold=args.fold,
        max_samples=args.max_samples,
        sample_rows=args.sample_rows,
        preset=args.preset,
    )

    descriptor_df = pd.DataFrame(descriptor_rows)
    token_df = pd.DataFrame(token_rows)
    similarity_df = pd.DataFrame(similarity_rows)
    grouped_df = _grouped_similarity_summary(similarity_df)

    descriptor_df.to_csv(args.output_dir / "descriptor_inventory.csv", index=False)
    token_df.to_csv(args.output_dir / "token_summary.csv", index=False)
    similarity_df.to_csv(args.output_dir / "pairwise_similarity.csv", index=False)
    grouped_df.to_csv(args.output_dir / "grouped_similarity_summary.csv", index=False)

    summary = {
        "datasets": args.datasets,
        "fold": args.fold,
        "preset": args.preset,
        "max_samples": args.max_samples,
        "sample_rows": args.sample_rows,
        "n_descriptor_rows": int(len(descriptor_df)),
        "n_similarity_rows": int(len(similarity_df)),
    }
    if not similarity_df.empty:
        summary["mean_cosine_same_family"] = float(similarity_df.loc[similarity_df["same_family"] == 1, "cosine_similarity"].mean())
        summary["mean_cosine_cross_family"] = float(similarity_df.loc[similarity_df["same_family"] == 0, "cosine_similarity"].mean())
        summary["mean_cosine_same_task_type"] = float(similarity_df.loc[similarity_df["same_task_type"] == 1, "cosine_similarity"].mean())
        summary["mean_cosine_cross_task_type"] = float(similarity_df.loc[similarity_df["same_task_type"] == 0, "cosine_similarity"].mean())
        summary["mean_cosine_anchor_anchor"] = float(
            similarity_df.loc[(similarity_df["is_anchor_a"] == 1) & (similarity_df["is_anchor_b"] == 1), "cosine_similarity"].mean()
        )
        summary["mean_cosine_subspace_subspace"] = float(
            similarity_df.loc[(similarity_df["family_a"] == "structural_subspace") & (similarity_df["family_b"] == "structural_subspace"), "cosine_similarity"].mean()
        )
        control_rows = grouped_df.loc[(grouped_df["same_dataset"] == 0)]
        if not control_rows.empty:
            summary["best_cross_dataset_group"] = control_rows.sort_values(
                by="mean_cosine_similarity", ascending=False
            ).iloc[0].to_dict()
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
