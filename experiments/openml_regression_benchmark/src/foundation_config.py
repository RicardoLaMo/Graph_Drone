from __future__ import annotations

from pathlib import Path


def prepare_foundation_config(
    *,
    source_config: Path,
    output_config: Path,
    data_path: str,
    seed: int,
    smoke: bool = False,
    amp: bool | None = None,
    cat_policy: str | None = None,
) -> Path:
    lines: list[str] = []
    in_data = False
    cat_policy_written = False

    for line in source_config.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            if in_data and cat_policy is not None and not cat_policy_written:
                lines.append(f'cat_policy = "{cat_policy}"')
                cat_policy_written = True
            in_data = stripped == "[data]"
            lines.append(line)
            continue

        if not in_data and stripped.startswith("seed = "):
            lines.append(f"seed = {seed}")
        elif in_data and stripped.startswith("path = "):
            lines.append(f'path = "{data_path}"')
        elif in_data and stripped.startswith("cat_policy = "):
            if cat_policy is None:
                lines.append(line)
            else:
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

    if in_data and cat_policy is not None and not cat_policy_written:
        lines.append(f'cat_policy = "{cat_policy}"')

    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text("\n".join(lines) + "\n")
    return output_config
