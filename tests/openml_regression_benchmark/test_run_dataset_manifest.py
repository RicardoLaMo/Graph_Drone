from __future__ import annotations

from experiments.openml_regression_benchmark.scripts.run_dataset_manifest import current_branch_name


def test_current_branch_name_prefers_branch(monkeypatch) -> None:
    outputs = iter(["codex/graphdrone-alpha-gate\n"])

    def fake_check_output(*_args, **_kwargs):
        return next(outputs)

    monkeypatch.setattr(
        "experiments.openml_regression_benchmark.scripts.run_dataset_manifest.subprocess.check_output",
        fake_check_output,
    )

    assert current_branch_name() == "codex/graphdrone-alpha-gate"


def test_current_branch_name_falls_back_to_commit(monkeypatch) -> None:
    outputs = iter(["\n", "161d664\n"])

    def fake_check_output(*_args, **_kwargs):
        return next(outputs)

    monkeypatch.setattr(
        "experiments.openml_regression_benchmark.scripts.run_dataset_manifest.subprocess.check_output",
        fake_check_output,
    )

    assert current_branch_name() == "161d664"
