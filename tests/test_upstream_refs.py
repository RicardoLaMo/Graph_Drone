from __future__ import annotations

from pathlib import Path

from experiments.tabr_california_baseline.src.upstream_refs import extract_upstream_california_refs


def test_extract_upstream_california_refs_reads_known_report_layout(tmp_path: Path):
    upstream_root = tmp_path / "tabr"
    report_dir = upstream_root / "exp" / "tabr" / "california" / "0-evaluation" / "0"
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text(
        '{"metrics":{"test":{"rmse":0.41,"mae":0.25,"r2":0.87}}}'
    )
    config_path = upstream_root / "exp" / "tabr" / "california" / "0-evaluation" / "0.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("seed = 0\ncontext_size = 96\n")

    df = extract_upstream_california_refs(upstream_root)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["family"] == "tabr"
    assert row["dataset"] == "california"
    assert row["config_name"] == "0-evaluation/0"
    assert row["rmse"] == 0.41
