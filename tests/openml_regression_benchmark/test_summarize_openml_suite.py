from __future__ import annotations

import json

from experiments.openml_regression_benchmark.scripts.summarize_openml_suite import discover_run_dirs


def test_discover_run_dirs_prefers_suite_manifest(tmp_path):
    reports_root = tmp_path / "reports"
    reports_root.mkdir()
    manifest = [
        {"dataset": "wine_quality", "repeat": 0, "fold": 1, "run_name": "wine_quality__r0f1", "smoke": False},
        {"dataset": "wine_quality", "repeat": 0, "fold": 1, "run_name": "wine_quality__r0f1", "smoke": False},
        {"dataset": "houses", "repeat": 0, "fold": 2, "run_name": "houses__r0f2", "smoke": False},
    ]
    (reports_root / "suite_manifest.json").write_text(json.dumps(manifest))

    run_dirs = discover_run_dirs(reports_root)

    assert run_dirs == [
        reports_root / "houses__r0f2",
        reports_root / "wine_quality__r0f1",
    ]
