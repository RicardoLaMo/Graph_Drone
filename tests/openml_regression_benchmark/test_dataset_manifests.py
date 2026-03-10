from pathlib import Path

from experiments.openml_regression_benchmark.src.dataset_manifests import (
    available_manifest_datasets,
    load_manifest_for_dataset,
    manifest_path_for_dataset,
)


def test_available_manifest_datasets_cover_all_benchmark_datasets() -> None:
    expected = {
        "airfoil_self_noise",
        "concrete_compressive_strength",
        "diamonds",
        "healthcare_insurance_expenses",
        "houses",
        "miami_housing",
        "used_fiat_500",
        "wine_quality",
    }
    assert expected == set(available_manifest_datasets())


def test_manifest_path_and_branch_metadata_are_stable() -> None:
    manifest = load_manifest_for_dataset("diamonds")
    assert manifest_path_for_dataset("diamonds").name == "diamonds.json"
    assert manifest.branch_name == "codex/graphdrone-diamonds"
    assert manifest.worktree_name == "graphdrone-diamonds"
    assert manifest.effective_graphdrone_gpu_span(exclusive_graphdrone=False) == 1
    assert manifest.effective_graphdrone_gpu_span(exclusive_graphdrone=True) == 4
    assert manifest.output_root_path == Path(manifest.output_root_path)
