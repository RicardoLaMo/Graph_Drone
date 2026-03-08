# M0 Environment

- Status: `PASS`
- Worktree: `feature/tabr-california-baseline`
- Machine: Apple Silicon macOS arm64
- Python: `3.12.12`
- PyTorch: `2.10.0`
- MPS: built `true`, available `true`
- CUDA: `false`
- FAISS: `1.13.2` via `faiss-cpu`

## Findings

- Upstream TabR is not ready for this machine as-is.
- Upstream environment pins older `pytorch=1.13.1` and `faiss-gpu=1.7.2`.
- Upstream device selection only supports `cpu` and `cuda`, not `mps`.
- CPU-first is the correct starting point for a trusted champion baseline on this machine.

## Compatibility Notes

- The TabR execution environment was isolated in `.venv-tabr312` using Python `3.12`.
- Two upstream compatibility shims were required in the inspected clone:
  - CPU FAISS search had to use NumPy arrays instead of `faiss.contrib.torch_utils`.
  - `torch.load(..., weights_only=False)` was needed for PyTorch `2.10`.

## Artifact

- Environment snapshot: [environment.json](/Volumes/MacMini/Projects/Graph_Drone/.worktrees/tabr-california-baseline/experiments/tabr_california_baseline/artifacts/environment.json)
