from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SHARED_PYTHON = REPO_ROOT / ".venv-h200" / "bin" / "python"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.openml_regression_benchmark.src.dataset_manifests import (  # noqa: E402
    available_manifest_datasets,
    load_manifest,
    load_manifest_for_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one OpenML dataset benchmark manifest")
    parser.add_argument("--dataset", choices=available_manifest_datasets())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--exclusive-graphdrone", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def build_command(
    *,
    manifest,
    smoke: bool,
    exclusive_graphdrone: bool,
    gpus_override: str,
    output_root: Path,
) -> list[str]:
    script_path = REPO_ROOT / "experiments" / "openml_regression_benchmark" / "scripts" / "run_openml_suite.py"
    cmd = [
        str(SHARED_PYTHON),
        "-c",
        f"import runpy; runpy.run_path({str(script_path)!r}, run_name='__main__')",
        "--datasets",
        manifest.dataset,
        "--repeat",
        str(manifest.repeat),
        "--seed",
        str(manifest.seed),
        "--split-seed",
        str(manifest.split_seed),
        "--models",
        *manifest.models,
        "--folds",
        *(str(fold) for fold in manifest.folds),
        "--graphdrone-max-train-samples",
        str(manifest.graphdrone_max_train_samples),
        "--tabpfn-max-train-samples",
        str(manifest.tabpfn_max_train_samples),
        "--gpus",
        gpus_override or manifest.gpus,
        "--gpu-order",
        manifest.gpu_order,
        "--graphdrone-gpu-span",
        str(manifest.effective_graphdrone_gpu_span(exclusive_graphdrone=exclusive_graphdrone)),
        "--graphdrone-parallel-workers",
        str(manifest.graphdrone_parallel_workers),
        "--max-concurrent-jobs",
        str(manifest.max_concurrent_jobs),
        "--output-root",
        str(output_root),
    ]
    if smoke:
        cmd.append("--smoke")
    return cmd


def current_branch_name() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "branch", "--show-current"],
        text=True,
    ).strip()


def main() -> None:
    args = parse_args()
    if bool(args.dataset) == bool(args.manifest):
        raise SystemExit("Provide exactly one of --dataset or --manifest")

    manifest = load_manifest(args.manifest) if args.manifest else load_manifest_for_dataset(args.dataset)
    output_root = args.output_root or manifest.output_root_path
    output_root.mkdir(parents=True, exist_ok=True)

    command = build_command(
        manifest=manifest,
        smoke=args.smoke,
        exclusive_graphdrone=args.exclusive_graphdrone,
        gpus_override=args.gpus,
        output_root=output_root,
    )
    meta = {
        "dataset": manifest.dataset,
        "branch_name": current_branch_name(),
        "worktree_name": REPO_ROOT.name,
        "notes": manifest.notes,
        "exclusive_graphdrone": args.exclusive_graphdrone,
        "smoke": args.smoke,
        "command": command,
    }
    (output_root / "dataset_manifest_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (output_root / "dataset_manifest_command.txt").write_text(" ".join(shlex.quote(part) for part in command) + "\n")

    if args.print_command:
        print(" ".join(shlex.quote(part) for part in command))
        return

    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
