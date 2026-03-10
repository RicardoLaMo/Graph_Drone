#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIFORNIA_OPENML_DID = 44024


def run_cmd(args: list[str]) -> dict[str, object]:
    proc = subprocess.run(args, cwd=REPO_ROOT, text=True, capture_output=True)
    return {
        "cmd": args,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def torch_smoke() -> dict[str, object]:
    import tabpfn
    import torch

    result: dict[str, object] = {
        "torch_version": torch.__version__,
        "tabpfn_version": getattr(tabpfn, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    device_names = []
    for idx in range(torch.cuda.device_count()):
        device_names.append(torch.cuda.get_device_name(idx))
    result["cuda_device_names"] = device_names

    if torch.cuda.is_available():
        x = torch.randn((1024, 1024), device="cuda")
        y = torch.randn((1024, 1024), device="cuda")
        z = x @ y
        result["matmul_norm"] = float(z.norm().item())
    else:
        x = torch.randn((256, 256))
        y = torch.randn((256, 256))
        z = x @ y
        result["cpu_matmul_norm"] = float(z.norm().item())
    return result


def git_smoke() -> dict[str, object]:
    remote = run_cmd(["git", "remote", "get-url", "origin"])
    remote_auth = run_cmd(["git", "ls-remote", "origin", "HEAD"])
    user_name = run_cmd(["git", "config", "--get", "user.name"])
    user_email = run_cmd(["git", "config", "--get", "user.email"])
    status = run_cmd(["git", "status", "--short", "--branch"])
    gh_path = shutil.which("gh")
    gh_status = run_cmd(["gh", "auth", "status"]) if gh_path else None
    ready = (
        bool(remote["stdout"])
        and remote_auth["returncode"] == 0
        and bool(user_name["stdout"])
        and bool(user_email["stdout"])
    )
    return {
        "remote_origin": remote["stdout"],
        "remote_auth": remote_auth,
        "user_name": user_name["stdout"],
        "user_email": user_email["stdout"],
        "status": status["stdout"],
        "gh_path": gh_path,
        "gh_auth": gh_status,
        "github_cli_ready": bool(gh_status and gh_status["returncode"] == 0),
        "ready": ready,
    }


def hf_smoke() -> dict[str, object]:
    try:
        import huggingface_hub as hf_hub
        from huggingface_hub import HfApi
    except Exception as exc:  # pragma: no cover - import failure is the signal
        return {"import_error": repr(exc), "ready": False}

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    offline_mode = os.environ.get("HF_HUB_OFFLINE")
    if offline_mode and offline_mode not in {"0", "false", "False"}:
        return {
            "huggingface_hub_version": getattr(hf_hub, "__version__", "unknown"),
            "cli_path": shutil.which("hf") or shutil.which("huggingface-cli"),
            "offline_mode": offline_mode,
            "token_env_present": bool(token),
            "auth": None,
            "ready": False,
        }

    if token:
        try:
            whoami = HfApi().whoami(token=token)
        except Exception as exc:
            return {
                "huggingface_hub_version": getattr(hf_hub, "__version__", "unknown"),
                "cli_path": shutil.which("hf") or shutil.which("huggingface-cli"),
                "offline_mode": offline_mode,
                "token_env_present": True,
                "auth_error": repr(exc),
                "ready": False,
            }
        return {
            "huggingface_hub_version": getattr(hf_hub, "__version__", "unknown"),
            "cli_path": shutil.which("hf") or shutil.which("huggingface-cli"),
            "offline_mode": offline_mode,
            "token_env_present": True,
            "auth": {
                "account": whoami.get("name") or whoami.get("fullname") or "ok",
                "auth_mode": "env_token",
            },
            "ready": True,
        }

    cli = shutil.which("hf") or shutil.which("huggingface-cli")
    if cli is None:
        return {
            "huggingface_hub_version": getattr(hf_hub, "__version__", "unknown"),
            "cli_path": None,
            "offline_mode": offline_mode,
            "token_env_present": False,
            "auth": None,
            "ready": False,
        }

    auth_cmd = [cli, "auth", "whoami"] if Path(cli).name == "hf" else [cli, "whoami"]
    auth = run_cmd(auth_cmd)
    return {
        "huggingface_hub_version": getattr(hf_hub, "__version__", "unknown"),
        "cli_path": cli,
        "offline_mode": offline_mode,
        "token_env_present": False,
        "auth": auth,
        "ready": auth["returncode"] == 0,
    }


def openml_smoke(dataset_id: int, download_data: bool) -> dict[str, object]:
    import openml

    try:
        dataset = openml.datasets.get_dataset(dataset_id, download_data=download_data)
    except TypeError:
        dataset = openml.datasets.get_dataset(dataset_id)

    result: dict[str, object] = {
        "openml_version": openml.__version__,
        "dataset_id": dataset_id,
        "dataset_name": getattr(dataset, "name", None),
        "default_target_attribute": getattr(dataset, "default_target_attribute", None),
    }

    if download_data:
        X, y, categorical, names = dataset.get_data(
            dataset_format="dataframe",
            target=dataset.default_target_attribute,
        )
        result["data_shape"] = [int(X.shape[0]), int(X.shape[1])]
        result["target_size"] = int(len(y))
        result["categorical_features"] = int(sum(bool(v) for v in categorical))
        result["feature_name_count"] = int(len(names))
    result["ready"] = True
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the P0 H200 runtime stack.")
    parser.add_argument("--torch-smoke", action="store_true")
    parser.add_argument("--git-smoke", action="store_true")
    parser.add_argument("--hf-smoke", action="store_true")
    parser.add_argument("--openml-smoke", action="store_true")
    parser.add_argument("--openml-dataset-id", type=int, default=DEFAULT_CALIFORNIA_OPENML_DID)
    parser.add_argument("--openml-download", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: dict[str, object] = {}
    failed = False

    if args.torch_smoke:
        try:
            results["torch"] = torch_smoke()
        except Exception as exc:
            results["torch_error"] = repr(exc)
            failed = True

    if args.git_smoke:
        try:
            results["git"] = git_smoke()
            failed = failed or not bool(results["git"].get("ready"))
        except Exception as exc:
            results["git_error"] = repr(exc)
            failed = True

    if args.hf_smoke:
        try:
            results["huggingface"] = hf_smoke()
            failed = failed or not bool(results["huggingface"].get("ready"))
        except Exception as exc:
            results["huggingface_error"] = repr(exc)
            failed = True

    if args.openml_smoke:
        try:
            results["openml"] = openml_smoke(args.openml_dataset_id, args.openml_download)
            failed = failed or not bool(results["openml"].get("ready"))
        except Exception as exc:
            results["openml_error"] = repr(exc)
            failed = True

    if not results:
        raise SystemExit("Pass at least one smoke flag.")

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(json.dumps(results, indent=2))

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
