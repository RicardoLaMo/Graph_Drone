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
from experiments.tab_foundation_compare.src.runtime_support import (
    ALIGNED_CANONICAL_SEED,
    default_foundation_python,
    default_single_gpu_cuda_visible_devices,
    default_tabr_upstream_root,
    seed_aware_dataset_name,
    seed_aware_run_name,
)
from experiments.tabr_california_baseline.src.run_config import prepare_eval_config
from experiments.tabr_california_baseline.src.upstream_refs import extract_upstream_california_refs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, default=default_tabr_upstream_root(REPO_ROOT))
    parser.add_argument(
        "--venv-python",
        type=Path,
        default=default_foundation_python(REPO_ROOT),
    )
    parser.add_argument(
        "--device-policy",
        choices=["auto", "cpu"],
        default="auto",
        help="Use CUDA when available in auto mode; cpu blanks CUDA visibility.",
    )
    parser.add_argument("--config", default="0-evaluation/0")
    parser.add_argument(
        "--seed",
        "--split-seed",
        dest="seed",
        type=int,
        default=ALIGNED_CANONICAL_SEED,
        help="Aligned California split seed. Keeps historical filenames when set to 42.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "experiments" / "tab_foundation_compare")
    parser.add_argument("--dataset-name", default="")
    return parser.parse_args()


def _patch_tabr_for_modern_env(upstream_root: Path) -> None:
    util_path = upstream_root / "lib" / "util.py"
    util_text = util_path.read_text()
    util_variants = [
        (
            "def load_checkpoint(output: str | Path, **kwargs) -> Any:\n"
            "    return torch.load(get_checkpoint_path(output), **kwargs)\n",
            "def load_checkpoint(output: str | Path, **kwargs) -> Any:\n"
            "    kwargs.setdefault('weights_only', False)\n"
            "    return torch.load(get_checkpoint_path(output), **kwargs)\n",
        ),
        (
            "def load_checkpoint(output: Union[str, Path], **kwargs) -> JSONDict:\n"
            "    return torch.load(get_checkpoint_path(output), **kwargs)\n",
            "def load_checkpoint(output: Union[str, Path], **kwargs) -> JSONDict:\n"
            "    kwargs.setdefault('weights_only', False)\n"
            "    return torch.load(get_checkpoint_path(output), **kwargs)\n",
        ),
    ]
    for util_old, util_new in util_variants:
        if util_old in util_text and util_new not in util_text:
            util_text = util_text.replace(util_old, util_new)
    util_path.write_text(util_text)

    data_path = upstream_root / "lib" / "data.py"
    data_text = data_path.read_text()
    onehot_old = "handle_unknown='ignore', sparse=False, dtype=np.float32"
    onehot_new = "handle_unknown='ignore', sparse_output=False, dtype=np.float32"
    if onehot_old in data_text and onehot_new not in data_text:
        data_text = data_text.replace(onehot_old, onehot_new)
        data_path.write_text(data_text)

    tabr_path = upstream_root / "bin" / "tabr.py"
    text = tabr_path.read_text()
    old_search = (
        "        with torch.no_grad():\n"
        "            if self.search_index is None:\n"
        "                self.search_index = (\n"
        "                    faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d_main)\n"
        "                    if device.type == 'cuda'\n"
        "                    else faiss.IndexFlatL2(d_main)\n"
        "                )\n"
        "            # Updating the index is much faster than creating a new one.\n"
        "            self.search_index.reset()\n"
        "            self.search_index.add(candidate_k)  # type: ignore[code]\n"
        "            distances: Tensor\n"
        "            context_idx: Tensor\n"
        "            distances, context_idx = self.search_index.search(  # type: ignore[code]\n"
        "                k, context_size + (1 if is_train else 0)\n"
        "            )\n"
        "            if is_train:\n"
        "                # NOTE: to avoid leakage, the index i must be removed from the i-th row,\n"
        "                # (because of how candidate_k is constructed).\n"
        "                distances[\n"
        "                    context_idx == torch.arange(batch_size, device=device)[:, None]\n"
        "                ] = torch.inf\n"
        "                # Not the most elegant solution to remove the argmax, but anyway.\n"
        "                context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])\n"
    )
    new_search = (
        "        with torch.no_grad():\n"
        "            use_gpu_faiss = device.type == 'cuda' and hasattr(faiss, 'StandardGpuResources')\n"
        "            if self.search_index is None:\n"
        "                self.search_index = (\n"
        "                    faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d_main)\n"
        "                    if use_gpu_faiss\n"
        "                    else faiss.IndexFlatL2(d_main)\n"
        "                )\n"
        "            # Updating the index is much faster than creating a new one.\n"
        "            self.search_index.reset()\n"
        "            distances: Tensor\n"
        "            context_idx: Tensor\n"
        "            if use_gpu_faiss:\n"
        "                self.search_index.add(candidate_k)  # type: ignore[code]\n"
        "                distances, context_idx = self.search_index.search(  # type: ignore[code]\n"
        "                    k, context_size + (1 if is_train else 0)\n"
        "                )\n"
        "            else:\n"
        "                candidate_k_np = np.ascontiguousarray(candidate_k.detach().cpu().numpy())\n"
        "                k_np = np.ascontiguousarray(k.detach().cpu().numpy())\n"
        "                self.search_index.add(candidate_k_np)  # type: ignore[code]\n"
        "                distances_np, context_idx_np = self.search_index.search(  # type: ignore[code]\n"
        "                    k_np, context_size + (1 if is_train else 0)\n"
        "                )\n"
        "                distances = torch.from_numpy(distances_np).to(device=device)\n"
        "                context_idx = torch.from_numpy(context_idx_np).to(device=device)\n"
        "            if is_train:\n"
        "                # NOTE: to avoid leakage, the index i must be removed from the i-th row,\n"
        "                # (because of how candidate_k is constructed).\n"
        "                distances[\n"
        "                    context_idx == torch.arange(batch_size, device=device)[:, None]\n"
        "                ] = torch.inf\n"
        "                # Not the most elegant solution to remove the argmax, but anyway.\n"
        "                context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])\n"
    )
    if old_search in text and "use_gpu_faiss" not in text:
        text = text.replace(old_search, new_search)
    tabr_path.write_text(text)


def main():
    args = parse_args()
    output_root = args.output_root.resolve()
    art_dir = output_root / "artifacts"
    log_dir = output_root / "logs"
    rep_dir = output_root / "reports"
    cfg_dir = output_root / "configs"
    for directory in [art_dir, log_dir, rep_dir, cfg_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset_name or seed_aware_dataset_name(
        "california_aligned_ours",
        args.seed,
        canonical_seed=ALIGNED_CANONICAL_SEED,
    )
    local_data_dir = art_dir / "data" / dataset_name
    write_aligned_california_dataset(local_data_dir, seed=args.seed)

    source_config = args.upstream_root / "exp" / "tabr" / "california" / f"{args.config}.toml"
    if not source_config.exists():
        raise FileNotFoundError(source_config)
    run_name = seed_aware_run_name(
        "tabr__" + args.config.replace("/", "__"),
        args.seed,
        canonical_seed=ALIGNED_CANONICAL_SEED,
        smoke=args.smoke,
    )
    local_config = cfg_dir / f"{run_name}.toml"
    prepare_eval_config(
        source_config=source_config,
        output_config=local_config,
        data_path=str(local_data_dir),
        smoke=args.smoke,
    )

    upstream_data_dir = args.upstream_root / "data" / dataset_name
    upstream_data_dir.parent.mkdir(parents=True, exist_ok=True)
    if upstream_data_dir.exists():
        shutil.rmtree(upstream_data_dir)
    shutil.copytree(local_data_dir, upstream_data_dir)

    _patch_tabr_for_modern_env(args.upstream_root)

    env = os.environ.copy()
    env["PROJECT_DIR"] = str(args.upstream_root)
    if args.device_policy == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = env.get(
            "CUDA_VISIBLE_DEVICES",
            default_single_gpu_cuda_visible_devices(),
        )
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("TQDM_DISABLE", "1")
    stdout_path = log_dir / f"{run_name}.stdout.log"
    stderr_path = log_dir / f"{run_name}.stderr.log"
    started_at = time.time()
    command = [str(args.venv_python), "bin/tabr.py", str(local_config), "--force"]
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
        "seed": args.seed,
        "dataset_name": dataset_name,
        "smoke": args.smoke,
        "device_policy": args.device_policy,
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
        report["best_epoch"] = upstream_report.get("best_epoch")
    report_path = rep_dir / f"{run_name}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
