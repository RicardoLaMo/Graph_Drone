from __future__ import annotations

from experiments.openml_classification_benchmark.src.foundation_config import prepare_foundation_config


SAMPLE_CONFIG = """
seed = 7

[data]
path = "/tmp/source"
num_policy = "quantile"
cat_policy = "ordinal"
y_policy = None

[model]
d_main = 64
""".strip()


def test_prepare_foundation_config_writes_null_token_for_tabr_numeric_only(tmp_path) -> None:
    source = tmp_path / "source.toml"
    target = tmp_path / "target.toml"
    source.write_text(SAMPLE_CONFIG + "\n")

    prepare_foundation_config(
        source_config=source,
        output_config=target,
        data_path="/tmp/new-data",
        seed=42,
        cat_policy=None,
        null_toml_token="__null__",
    )

    text = target.read_text()
    assert 'path = "/tmp/new-data"' in text
    assert "seed = 42" in text
    assert 'cat_policy = "__null__"' in text
    assert 'cat_policy = "ordinal"' not in text


def test_prepare_foundation_config_drops_cat_policy_when_not_needed(tmp_path) -> None:
    source = tmp_path / "source.toml"
    target = tmp_path / "target.toml"
    source.write_text(SAMPLE_CONFIG + "\n")

    prepare_foundation_config(
        source_config=source,
        output_config=target,
        data_path="/tmp/new-data",
        seed=42,
        cat_policy=None,
        null_toml_token=None,
    )

    text = target.read_text()
    assert 'path = "/tmp/new-data"' in text
    assert "seed = 42" in text
    assert "cat_policy" not in text


def test_prepare_foundation_config_writes_null_num_policy_for_categorical_only_tabr(tmp_path) -> None:
    source = tmp_path / "source.toml"
    target = tmp_path / "target.toml"
    source.write_text(SAMPLE_CONFIG + "\n")

    prepare_foundation_config(
        source_config=source,
        output_config=target,
        data_path="/tmp/new-data",
        seed=42,
        num_policy=None,
        cat_policy="ordinal",
        null_toml_token="__null__",
    )

    text = target.read_text()
    assert 'num_policy = "__null__"' in text
    assert 'cat_policy = "ordinal"' in text
