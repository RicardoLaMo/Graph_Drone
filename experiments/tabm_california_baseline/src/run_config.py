from __future__ import annotations

from pathlib import Path


def prepare_model_config(
    *,
    source_config: Path,
    output_config: Path,
    data_path: str,
    smoke: bool = False,
    amp: bool = False,
) -> Path:
    lines = []
    for line in source_config.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("path = "):
            lines.append(f'path = "{data_path}"')
        elif stripped.startswith("amp = "):
            lines.append(f"amp = {'true' if amp else 'false'}")
        elif smoke and stripped.startswith("n_epochs = "):
            lines.append("n_epochs = 3")
        elif smoke and stripped.startswith("patience = "):
            lines.append("patience = 2")
        else:
            lines.append(line)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text("\n".join(lines) + "\n")
    return output_config
