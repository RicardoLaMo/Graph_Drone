# TabArena Bridge

This folder is a lightweight bridge between Graph_Drone experiments and the TabArena benchmark style.

It does **not** replace the existing experiment code.

It provides:

- a dataset manifest tailored to Graph_Drone
- a canonical benchmark matrix
- scripts to validate and render that matrix

The purpose is to make local protocol decisions firm before moving the heavier runs to the H200 environment.

## Files

- `configs/benchmark_manifest.json`
- `scripts/render_benchmark_plan.py`
- `artifacts/benchmark_matrix.csv` (generated)
- `artifacts/benchmark_matrix.md` (generated)

## Usage

```bash
source .venv/bin/activate
python experiments/tabarena_bridge/scripts/render_benchmark_plan.py
```

This writes the current benchmark matrix under `artifacts/`.
