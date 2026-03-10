from __future__ import annotations

from experiments.tabpfn_view_router.src.runtime import (
    assign_view_devices,
    build_device_plan,
    infer_parallel_workers,
    resolve_device_spec,
)


def test_resolve_device_spec_splits_comma_separated_devices() -> None:
    assert resolve_device_spec("cuda:0, cuda:1") == ["cuda:0", "cuda:1"]


def test_assign_view_devices_round_robins_in_per_view_mode() -> None:
    plan = assign_view_devices(
        ["FULL", "GEO", "SOCIO", "LOWRANK"],
        ["cuda:0", "cuda:1"],
        device_mode="per_view",
    )
    assert plan == {
        "FULL": "cuda:0",
        "GEO": "cuda:1",
        "SOCIO": "cuda:0",
        "LOWRANK": "cuda:1",
    }


def test_assign_view_devices_keeps_full_pool_in_per_model_mode() -> None:
    plan = assign_view_devices(
        ["FULL", "GEO"],
        ["cuda:0", "cuda:1"],
        device_mode="per_model",
    )
    assert plan == {
        "FULL": ["cuda:0", "cuda:1"],
        "GEO": ["cuda:0", "cuda:1"],
    }


def test_infer_parallel_workers_uses_unique_cuda_devices_only() -> None:
    workers = infer_parallel_workers(
        {"FULL": "cuda:0", "GEO": "cuda:1", "SOCIO": "cuda:0"},
        device_mode="per_view",
        requested_parallel_workers=0,
    )
    assert workers == 2


def test_build_device_plan_caps_parallel_workers_for_per_model_mode() -> None:
    plan = build_device_plan(
        ["FULL", "GEO"],
        requested_device="cuda:0,cuda:1",
        device_mode="per_model",
        all_gpus=False,
        parallel_workers=8,
    )
    assert plan.parallel_workers == 1
    assert plan.view_devices["FULL"] == ["cuda:0", "cuda:1"]
