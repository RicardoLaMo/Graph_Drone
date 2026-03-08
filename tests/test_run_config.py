from __future__ import annotations

from pathlib import Path

from experiments.tabr_california_baseline.src.run_config import prepare_eval_config


def test_prepare_eval_config_rewrites_data_path_and_epochs(tmp_path: Path):
    source = tmp_path / "source.toml"
    source.write_text(
        "\n".join(
            [
                "seed = 0",
                "batch_size = 256",
                "patience = 16",
                "n_epochs = inf",
                "",
                "[data]",
                'path = ":data/california"',
            ]
        )
        + "\n"
    )
    output = tmp_path / "patched.toml"

    prepare_eval_config(
        source_config=source,
        output_config=output,
        data_path=":data/california_local",
        smoke=True,
    )

    text = output.read_text()
    assert 'path = ":data/california_local"' in text
    assert "n_epochs = 3" in text
