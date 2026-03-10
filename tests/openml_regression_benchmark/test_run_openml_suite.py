from collections import deque

from experiments.openml_regression_benchmark.scripts.run_openml_suite import (
    discover_free_gpus,
    gpu_span_for_model,
    ordered_gpu_indices,
    parse_gpu_indices,
    parse_gpu_status,
    pop_first_schedulable_task,
    resolve_gpu_pool,
    take_gpu_allocation,
)


def test_parse_gpu_indices_deduplicates_preserving_order():
    assert parse_gpu_indices("7,6,7,5") == [7, 6, 5]


def test_ordered_gpu_indices_supports_high_and_low_first():
    assert ordered_gpu_indices([1, 7, 3], gpu_order="high-first") == [7, 3, 1]
    assert ordered_gpu_indices([1, 7, 3], gpu_order="low-first") == [1, 3, 7]


def test_parse_gpu_status_reads_nvidia_smi_csv():
    status = parse_gpu_status("7, 1024, 0\n6, 8192, 85\n")
    assert status == [
        {"index": 7, "memory_used_mib": 1024, "utilization_gpu": 0},
        {"index": 6, "memory_used_mib": 8192, "utilization_gpu": 85},
    ]


def test_resolve_gpu_pool_prefers_env_pool_for_auto():
    gpu_status = [
        {"index": 0, "memory_used_mib": 0, "utilization_gpu": 0},
        {"index": 7, "memory_used_mib": 0, "utilization_gpu": 0},
    ]
    assert resolve_gpu_pool(
        gpu_arg="auto",
        env_gpu_pool="7,6,5,4,3,2,1,0",
        gpu_order="high-first",
        gpu_status=gpu_status,
    ) == [7, 6, 5, 4, 3, 2, 1, 0]


def test_discover_free_gpus_filters_on_thresholds():
    gpu_status = [
        {"index": 7, "memory_used_mib": 1024, "utilization_gpu": 0},
        {"index": 6, "memory_used_mib": 4096, "utilization_gpu": 10},
        {"index": 5, "memory_used_mib": 4097, "utilization_gpu": 0},
        {"index": 4, "memory_used_mib": 1024, "utilization_gpu": 11},
    ]
    assert discover_free_gpus(
        gpu_pool=[7, 6, 5, 4],
        gpu_status=gpu_status,
        gpu_order="high-first",
        memory_free_threshold_mib=4096,
        util_free_threshold=10,
    ) == [7, 6]


def test_graphdrone_gpu_span_is_clipped_to_four():
    assert gpu_span_for_model("GraphDrone", graphdrone_gpu_span=8) == 4
    assert gpu_span_for_model("TabPFN", graphdrone_gpu_span=8) == 1


def test_take_gpu_allocation_returns_contiguous_slice():
    available = [7, 6, 5, 4]
    allocation = take_gpu_allocation(available, span=2)
    assert allocation == (7, 6)
    assert available == [5, 4]
    assert take_gpu_allocation(available, span=3) is None


def test_pop_first_schedulable_task_skips_blocking_wide_job():
    tasks = deque(
        [
            {"dataset": "diamonds", "fold": 0, "model": "GraphDrone"},
            {"dataset": "houses", "fold": 0, "model": "TabPFN"},
        ]
    )

    task = pop_first_schedulable_task(tasks, available_gpu_count=1, graphdrone_gpu_span=4)

    assert task == {"dataset": "houses", "fold": 0, "model": "TabPFN"}
    assert list(tasks) == [{"dataset": "diamonds", "fold": 0, "model": "GraphDrone"}]


def test_pop_first_schedulable_task_preserves_queue_when_nothing_fits():
    tasks = deque([{"dataset": "diamonds", "fold": 0, "model": "GraphDrone"}])

    task = pop_first_schedulable_task(tasks, available_gpu_count=0, graphdrone_gpu_span=4)

    assert task is None
    assert list(tasks) == [{"dataset": "diamonds", "fold": 0, "model": "GraphDrone"}]
