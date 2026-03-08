from __future__ import annotations

from pathlib import Path


def prepare_eval_config(
    *,
    source_config: Path,
    output_config: Path,
    data_path: str,
    smoke: bool = False,
) -> Path:
    text = source_config.read_text()
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("path = "):
            lines.append(f'path = "{data_path}"')
        elif smoke and line.strip().startswith("n_epochs = "):
            lines.append("n_epochs = 3")
        elif smoke and line.strip().startswith("patience = "):
            lines.append("patience = 2")
        else:
            lines.append(line)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text("\n".join(lines) + "\n")
    return output_config

