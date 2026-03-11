from __future__ import annotations

from pathlib import Path


ALIGNED_CANONICAL_SEED = 42
UPSTREAM_CANONICAL_SEED = 0


def shared_h200_python(repo_root: Path) -> Path:
    root = repo_root.parent.parent if repo_root.parent.name == ".worktrees" else repo_root
    return root / ".venv-h200" / "bin" / "python"


def repository_root(repo_root: Path) -> Path:
    return repo_root.parent.parent if repo_root.parent.name == ".worktrees" else repo_root


def resolve_upstream_root(repo_root: Path, *suffix: str) -> Path:
    root = repository_root(repo_root)
    direct = root.joinpath(".external", *suffix)
    if direct.exists():
        return direct
    worktrees_root = root / ".worktrees"
    if worktrees_root.exists():
        for sibling in sorted(worktrees_root.iterdir()):
            candidate = sibling.joinpath(".external", *suffix)
            if candidate.exists():
                return candidate
    return direct


def default_foundation_python(repo_root: Path) -> Path:
    shared_python = shared_h200_python(repo_root)
    if shared_python.exists():
        return shared_python
    return repo_root / ".venv-foundation312" / "bin" / "python"


def default_tabr_upstream_root(repo_root: Path) -> Path:
    return resolve_upstream_root(repo_root, "tabr")


def default_tabm_upstream_root(repo_root: Path) -> Path:
    return resolve_upstream_root(repo_root, "tabm", "paper")


def default_single_gpu_cuda_visible_devices() -> str:
    return "7"


def seed_aware_dataset_name(
    base_name: str,
    seed: int,
    *,
    canonical_seed: int,
) -> str:
    if seed == canonical_seed:
        return base_name
    return f"{base_name}_seed{seed}"


def seed_aware_run_name(
    base_name: str,
    seed: int,
    *,
    canonical_seed: int,
    smoke: bool,
) -> str:
    name = base_name if seed == canonical_seed else f"{base_name}__seed{seed}"
    if smoke:
        name = f"{name}__smoke"
    return name
