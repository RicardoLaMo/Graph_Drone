from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS_DIR = REPO_ROOT / "experiments" / "openml_regression_benchmark" / "manifests"


@dataclass(frozen=True)
class DatasetRunManifest:
    dataset: str
    worktree_slug: str
    output_root: str
    models: tuple[str, ...]
    folds: tuple[int, ...]
    repeat: int
    seed: int
    split_seed: int
    graphdrone_max_train_samples: int
    tabpfn_max_train_samples: int
    graphdrone_gpu_span: int
    graphdrone_exclusive_gpu_span: int
    graphdrone_parallel_workers: int
    max_concurrent_jobs: int
    gpus: str
    gpu_order: str
    notes: str

    @property
    def branch_name(self) -> str:
        return f"codex/graphdrone-{self.worktree_slug}"

    @property
    def worktree_name(self) -> str:
        return f"graphdrone-{self.worktree_slug}"

    @property
    def output_root_path(self) -> Path:
        return REPO_ROOT / self.output_root

    def effective_graphdrone_gpu_span(self, *, exclusive_graphdrone: bool) -> int:
        if exclusive_graphdrone:
            return max(1, self.graphdrone_exclusive_gpu_span)
        return max(1, self.graphdrone_gpu_span)


def manifest_path_for_dataset(dataset: str) -> Path:
    return MANIFESTS_DIR / f"{dataset}.json"


def available_manifest_datasets() -> list[str]:
    return sorted(path.stem for path in MANIFESTS_DIR.glob("*.json"))


def load_manifest(path: Path) -> DatasetRunManifest:
    payload = json.loads(path.read_text())
    return DatasetRunManifest(
        dataset=str(payload["dataset"]),
        worktree_slug=str(payload["worktree_slug"]),
        output_root=str(payload["output_root"]),
        models=tuple(str(item) for item in payload["models"]),
        folds=tuple(int(item) for item in payload["folds"]),
        repeat=int(payload.get("repeat", 0)),
        seed=int(payload.get("seed", 42)),
        split_seed=int(payload.get("split_seed", 42)),
        graphdrone_max_train_samples=int(payload.get("graphdrone_max_train_samples", 0)),
        tabpfn_max_train_samples=int(payload.get("tabpfn_max_train_samples", 0)),
        graphdrone_gpu_span=int(payload.get("graphdrone_gpu_span", 1)),
        graphdrone_exclusive_gpu_span=int(payload.get("graphdrone_exclusive_gpu_span", payload.get("graphdrone_gpu_span", 1))),
        graphdrone_parallel_workers=int(payload.get("graphdrone_parallel_workers", 0)),
        max_concurrent_jobs=int(payload.get("max_concurrent_jobs", 8)),
        gpus=str(payload.get("gpus", "auto")),
        gpu_order=str(payload.get("gpu_order", "high-first")),
        notes=str(payload.get("notes", "")),
    )


def load_manifest_for_dataset(dataset: str) -> DatasetRunManifest:
    return load_manifest(manifest_path_for_dataset(dataset))
