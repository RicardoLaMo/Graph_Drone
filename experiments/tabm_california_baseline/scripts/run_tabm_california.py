from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.tab_foundation_compare.src.aligned_california import write_aligned_california_dataset
from experiments.tabm_california_baseline.src.run_config import prepare_model_config
from experiments.tabm_california_baseline.src.upstream_refs import extract_upstream_california_refs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, default=Path("/private/tmp/tabm_clone_inspect_20260308/paper"))
    parser.add_argument(
        "--venv-python",
        type=Path,
        default=REPO_ROOT / ".venv-foundation312" / "bin" / "python",
    )
    parser.add_argument("--config", default="0-evaluation/0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "experiments" / "tabm_california_baseline")
    parser.add_argument("--dataset-name", default="california_aligned_ours")
    return parser.parse_args()


def _patch_tabm_for_modern_torch(upstream_root: Path) -> None:
    util_path = upstream_root / "lib" / "util.py"
    text = util_path.read_text()
    old = "def load_checkpoint(output: str | Path, **kwargs) -> Any:\n    return _torch().load(get_checkpoint_path(output), **kwargs)\n"
    new = (
        "def load_checkpoint(output: str | Path, **kwargs) -> Any:\n"
        "    kwargs.setdefault('weights_only', False)\n"
        "    return _torch().load(get_checkpoint_path(output), **kwargs)\n"
    )
    if old in text and new not in text:
        util_path.write_text(text.replace(old, new))


def main():
    args = parse_args()
    output_root = args.output_root.resolve()
    art_dir = output_root / "artifacts"
    log_dir = output_root / "logs"
    rep_dir = output_root / "reports"
    cfg_dir = output_root / "configs"
    for directory in [art_dir, log_dir, rep_dir, cfg_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    local_data_dir = art_dir / "data" / args.dataset_name
    write_aligned_california_dataset(local_data_dir, seed=42)

    source_config = args.upstream_root / "exp" / "tabm" / "california" / f"{args.config}.toml"
    if not source_config.exists():
        raise FileNotFoundError(source_config)

    run_name = args.config.replace("/", "__") + ("__smoke" if args.smoke else "")
    local_config = cfg_dir / f"{run_name}.toml"
    prepare_model_config(
        source_config=source_config,
        output_config=local_config,
        data_path=f"data/{args.dataset_name}",
        smoke=args.smoke,
        amp=False,
    )

    upstream_data_dir = args.upstream_root / "data" / args.dataset_name
    upstream_data_dir.parent.mkdir(parents=True, exist_ok=True)
    if upstream_data_dir.exists():
        shutil.rmtree(upstream_data_dir)
    shutil.copytree(local_data_dir, upstream_data_dir)

    _patch_tabm_for_modern_torch(args.upstream_root)

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    stdout_path = log_dir / f"{run_name}.stdout.log"
    stderr_path = log_dir / f"{run_name}.stderr.log"
    started_at = time.time()
    command = [str(args.venv_python), "bin/model.py", str(local_config), "--force"]
    with stdout_path.open("w") as stdout_handle, stderr_path.open("w") as stderr_handle:
        result = subprocess.run(
            command,
            cwd=args.upstream_root,
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )

    refs = extract_upstream_california_refs(args.upstream_root)
    refs.to_csv(art_dir / "upstream_reference_metrics.csv", index=False)

    run_output_dir = local_config.with_suffix("")
    summary_path = run_output_dir / "summary.json"
    upstream_report_path = run_output_dir / "report.json"
    report = {
        "config": args.config,
        "smoke": args.smoke,
        "returncode": result.returncode,
        "local_config": str(local_config),
        "local_data_dir": str(local_data_dir),
        "run_output_dir": str(run_output_dir),
        "upstream_report_path": str(upstream_report_path),
        "summary_path": str(summary_path),
        "duration_seconds": round(time.time() - started_at, 3),
    }
    if summary_path.exists():
        report["summary"] = json.loads(summary_path.read_text())
    if upstream_report_path.exists():
        upstream_report = json.loads(upstream_report_path.read_text())
        report["metrics"] = upstream_report.get("metrics")
        report["best_step"] = upstream_report.get("best_step")
    report_path = rep_dir / f"{run_name}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
