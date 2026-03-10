from __future__ import annotations

from pathlib import Path

from experiments.openml_regression_benchmark.scripts.run_tabr_openml import resolve_upstream_report_path


def test_resolve_upstream_report_path_prefers_expected_location(tmp_path: Path) -> None:
    run_output_dir = tmp_path / "tabr_config"
    run_output_dir.mkdir()
    expected = run_output_dir / "report.json"
    expected.write_text("{}\n")

    assert resolve_upstream_report_path(run_output_dir) == expected


def test_resolve_upstream_report_path_falls_back_to_matching_directory(tmp_path: Path) -> None:
    run_output_dir = tmp_path / "tabr_config"
    fallback_dir = tmp_path / "tabr_config_prev"
    fallback_dir.mkdir()
    fallback = fallback_dir / "report.json"
    fallback.write_text("{}\n")

    assert resolve_upstream_report_path(run_output_dir) == fallback


def test_resolve_upstream_report_path_returns_none_when_absent(tmp_path: Path) -> None:
    run_output_dir = tmp_path / "tabr_config"

    assert resolve_upstream_report_path(run_output_dir) is None
