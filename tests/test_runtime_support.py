from pathlib import Path

from experiments.tab_foundation_compare.src.runtime_support import (
    ALIGNED_CANONICAL_SEED,
    UPSTREAM_CANONICAL_SEED,
    default_foundation_python,
    default_single_gpu_cuda_visible_devices,
    default_tabr_upstream_root,
    default_tabm_upstream_root,
    seed_aware_dataset_name,
    seed_aware_run_name,
    shared_h200_python,
)


def test_default_runtime_paths_are_repo_local_when_no_shared_env_exists():
    repo_root = Path("/tmp/foundation")
    assert default_foundation_python(repo_root) == repo_root / ".venv-foundation312" / "bin" / "python"
    assert default_tabr_upstream_root(repo_root) == repo_root / ".external" / "tabr"
    assert default_tabm_upstream_root(repo_root) == repo_root / ".external" / "tabm" / "paper"
    assert shared_h200_python(repo_root) == Path("/tmp/foundation/.venv-h200/bin/python")
    assert default_single_gpu_cuda_visible_devices() == "7"


def test_default_foundation_python_prefers_shared_h200_env(tmp_path: Path):
    repo_root = tmp_path / ".worktrees" / "tab-foundation-baseline"
    shared_python = tmp_path / ".venv-h200" / "bin" / "python"
    shared_python.parent.mkdir(parents=True)
    shared_python.write_text("")
    repo_root.mkdir(parents=True)
    assert default_foundation_python(repo_root) == shared_python
    assert default_tabr_upstream_root(repo_root) == tmp_path / ".external" / "tabr"
    assert default_tabm_upstream_root(repo_root) == tmp_path / ".external" / "tabm" / "paper"


def test_default_upstream_roots_fall_back_to_sibling_worktree_external(tmp_path: Path):
    repo_root = tmp_path / ".worktrees" / "graphdrone-phase"
    sibling_tabr = tmp_path / ".worktrees" / "tab-foundation-baseline" / ".external" / "tabr"
    sibling_tabm = tmp_path / ".worktrees" / "tab-foundation-baseline" / ".external" / "tabm" / "paper"
    sibling_tabr.mkdir(parents=True)
    sibling_tabm.mkdir(parents=True)
    repo_root.mkdir(parents=True)
    assert default_tabr_upstream_root(repo_root) == sibling_tabr
    assert default_tabm_upstream_root(repo_root) == sibling_tabm


def test_seed_aware_dataset_name_preserves_canonical_seed():
    assert seed_aware_dataset_name(
        "california_aligned_ours",
        ALIGNED_CANONICAL_SEED,
        canonical_seed=ALIGNED_CANONICAL_SEED,
    ) == "california_aligned_ours"
    assert seed_aware_dataset_name(
        "california_aligned_ours",
        43,
        canonical_seed=ALIGNED_CANONICAL_SEED,
    ) == "california_aligned_ours_seed43"


def test_seed_aware_run_name_preserves_historical_names():
    assert seed_aware_run_name(
        "tabr__0-evaluation__0",
        ALIGNED_CANONICAL_SEED,
        canonical_seed=ALIGNED_CANONICAL_SEED,
        smoke=False,
    ) == "tabr__0-evaluation__0"
    assert seed_aware_run_name(
        "tabr__0-evaluation__0",
        43,
        canonical_seed=ALIGNED_CANONICAL_SEED,
        smoke=False,
    ) == "tabr__0-evaluation__0__seed43"
    assert seed_aware_run_name(
        "0-evaluation__0",
        UPSTREAM_CANONICAL_SEED,
        canonical_seed=UPSTREAM_CANONICAL_SEED,
        smoke=True,
    ) == "0-evaluation__0__smoke"
    assert seed_aware_run_name(
        "0-evaluation__0",
        1,
        canonical_seed=UPSTREAM_CANONICAL_SEED,
        smoke=True,
    ) == "0-evaluation__0__seed1__smoke"
