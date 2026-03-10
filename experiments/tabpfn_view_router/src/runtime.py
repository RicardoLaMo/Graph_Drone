from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


DeviceSpec = str | list[str]


@dataclass(frozen=True)
class DevicePlan:
    requested_device: str
    resolved_device: DeviceSpec
    device_mode: str
    view_devices: dict[str, DeviceSpec]
    parallel_workers: int


def _split_device_list(device_arg: str) -> list[str]:
    return [part.strip() for part in device_arg.split(",") if part.strip()]


def serialize_device_spec(device_spec: DeviceSpec) -> str | list[str]:
    if isinstance(device_spec, list):
        return list(device_spec)
    return device_spec


def resolve_device_spec(device_arg: str, *, all_gpus: bool = False) -> DeviceSpec:
    device_arg = device_arg.strip() or "auto"

    if all_gpus and torch.cuda.is_available():
        return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]

    devices = _split_device_list(device_arg)
    if len(devices) > 1:
        return devices
    return devices[0] if devices else device_arg


def assign_view_devices(
    view_names: Sequence[str],
    device_spec: DeviceSpec,
    *,
    device_mode: str,
) -> dict[str, DeviceSpec]:
    if device_mode not in {"per_view", "per_model"}:
        raise ValueError(f"Unsupported device_mode={device_mode!r}")

    if isinstance(device_spec, list):
        if device_mode == "per_model":
            return {name: list(device_spec) for name in view_names}
        return {
            name: device_spec[idx % len(device_spec)]
            for idx, name in enumerate(view_names)
        }

    return {name: device_spec for name in view_names}


def infer_parallel_workers(
    view_devices: dict[str, DeviceSpec],
    *,
    device_mode: str,
    requested_parallel_workers: int,
) -> int:
    if device_mode == "per_model":
        return 1

    unique_devices = {
        spec
        for spec in view_devices.values()
        if isinstance(spec, str) and spec not in {"auto", "cpu", "mps"}
    }
    max_parallel = max(1, len(unique_devices))
    if requested_parallel_workers <= 0:
        return max_parallel
    return min(requested_parallel_workers, max_parallel)


def build_device_plan(
    view_names: Sequence[str],
    *,
    requested_device: str,
    device_mode: str,
    all_gpus: bool,
    parallel_workers: int,
) -> DevicePlan:
    resolved = resolve_device_spec(requested_device, all_gpus=all_gpus)
    view_devices = assign_view_devices(view_names, resolved, device_mode=device_mode)
    actual_workers = infer_parallel_workers(
        view_devices,
        device_mode=device_mode,
        requested_parallel_workers=parallel_workers,
    )
    return DevicePlan(
        requested_device=requested_device,
        resolved_device=resolved,
        device_mode=device_mode,
        view_devices=view_devices,
        parallel_workers=actual_workers,
    )
