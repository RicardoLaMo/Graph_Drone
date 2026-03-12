from __future__ import annotations

from pathlib import Path

from experiments.openml_classification_benchmark.src.openml_tasks import PreparedOpenMLClassificationSplit

_KEEP_POLICY = "__keep__"


def prepare_foundation_config(
    *,
    source_config: Path,
    output_config: Path,
    data_path: str,
    seed: int,
    smoke: bool = False,
    amp: bool | None = None,
    num_policy: str | None = _KEEP_POLICY,
    cat_policy: str | None = _KEEP_POLICY,
    null_toml_token: str | None = None,
) -> Path:
    lines: list[str] = []
    in_data = False
    num_policy_written = False
    cat_policy_written = False
    for line in source_config.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_data = stripped == "[data]"
            lines.append(line)
            continue
        if not in_data and stripped.startswith("seed = "):
            lines.append(f"seed = {seed}")
        elif in_data and stripped.startswith("path = "):
            lines.append(f'path = "{data_path}"')
        elif in_data and stripped.startswith("num_policy = "):
            if num_policy == _KEEP_POLICY:
                lines.append(line)
                num_policy_written = True
                continue
            if num_policy is None:
                if null_toml_token is not None:
                    lines.append(f'num_policy = "{null_toml_token}"')
                    num_policy_written = True
                continue
            lines.append(f'num_policy = "{num_policy}"')
            num_policy_written = True
        elif in_data and stripped.startswith("cat_policy = "):
            if cat_policy == _KEEP_POLICY:
                lines.append(line)
                cat_policy_written = True
                continue
            if cat_policy is None:
                if null_toml_token is not None:
                    lines.append(f'cat_policy = "{null_toml_token}"')
                    cat_policy_written = True
                continue
            lines.append(f'cat_policy = "{cat_policy}"')
            cat_policy_written = True
        elif amp is not None and stripped.startswith("amp = "):
            lines.append(f"amp = {'true' if amp else 'false'}")
        elif smoke and stripped.startswith("n_epochs = "):
            lines.append("n_epochs = 3")
        elif smoke and stripped.startswith("patience = "):
            lines.append("patience = 2")
        else:
            lines.append(line)

    output_config.parent.mkdir(parents=True, exist_ok=True)
    if null_toml_token is not None:
        if not num_policy_written:
            lines = _inject_data_key(lines, key="num_policy", value=f'"{null_toml_token}"')
        if not cat_policy_written:
            lines = _inject_data_key(lines, key="cat_policy", value=f'"{null_toml_token}"')
    output_config.write_text("\n".join(lines) + "\n")
    return output_config


def _inject_data_key(lines: list[str], *, key: str, value: str) -> list[str]:
    injected: list[str] = []
    in_data = False
    inserted = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            if in_data and not inserted:
                injected.append(f"{key} = {value}")
                inserted = True
            in_data = stripped == "[data]"
            injected.append(line)
        else:
            injected.append(line)
    if in_data and not inserted:
        injected.append(f"{key} = {value}")
    return injected


def resolve_tabr_config_name(split: PreparedOpenMLClassificationSplit) -> str:
    if len(split.class_labels) > 2:
        return "covtype/0-evaluation/0"
    if split.X_cat_train is not None:
        return "adult/0-evaluation/0"
    return "churn/0-evaluation/0"


def resolve_tabm_config_name(split: PreparedOpenMLClassificationSplit) -> str:
    if len(split.class_labels) > 2:
        return "mlp/covtype2/0-evaluation/0"
    if split.X_cat_train is not None:
        return "mlp/adult/0-evaluation/0"
    return "mlp/churn/0-evaluation/0"


def extract_foundation_metrics(
    upstream_metrics: dict[str, object],
    *,
    split_name: str,
) -> dict[str, float | None]:
    section = upstream_metrics[split_name]
    macro = section.get("macro avg", {}) if isinstance(section, dict) else {}
    return {
        "accuracy": float(section["accuracy"]),
        "macro_f1": float(macro["f1-score"]) if "f1-score" in macro else None,
        "roc_auc": float(section["roc-auc"]) if "roc-auc" in section else None,
        "pr_auc": None,
        "log_loss": float(section["cross-entropy"]) if "cross-entropy" in section else None,
    }


__all__ = [
    "extract_foundation_metrics",
    "prepare_foundation_config",
    "resolve_tabr_config_name",
    "resolve_tabm_config_name",
]
