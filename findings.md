# Findings

## 2026-03-08

### Branch state

- New comparison worktree created from `feature/tabr-california-baseline`.
- Branch: `feature/california-tab-foundation-compare`
- Inherited TabR tests pass in the new worktree:
  - `pytest tests/test_data_bridge.py tests/test_upstream_refs.py tests/test_run_config.py -q`
  - Result: `3 passed`

### TabR status carried forward

- Local fixed-config TabR California run is already validated on Apple Silicon CPU.
- Best local fixed result:
  - config `0-evaluation/0`
  - test RMSE `0.3949403615264023`
  - val RMSE `0.40094805430544206`
  - best epoch `49`
- This is strong enough to treat `TabR_CPU_fixed` as the current champion candidate.

### TabM upstream inspection

- Repo inspected from `https://github.com/yandex-research/tabm`
- Temp checkout:
  - `/private/tmp/tabm_clone_inspect_20260308`
- Commit inspected:
  - `28e47ae3`

### TabM structure

- `README.md` describes the modern `tabm` package for practical use.
- `paper/README.md` describes the experiment reproduction stack.
- This mirrors TabR's split between practical package and paper reproduction, but TabM's package is newer and cleaner.

### TabM environment fit

- Package dependency range is modern:
  - `torch>=1.12,<3`
  - `requires-python >=3.9`
- This is materially easier to fit into the Apple/PyTorch 2.x environment than upstream TabR.
- The full paper reproduction still expects its own `paper/` workflow and downloaded `data.tar` bundle.

### TabM California evidence

- Published California base TabM config:
  - `paper/exp/tabm/california/0-evaluation/0.toml`
- One shipped report:
  - `paper/exp/tabm/california/0-evaluation/0/report.json`
- Shipped seed-0 test RMSE from that report:
  - `0.4417858517027938`
- The paper-wide mean shown in `paper/README.md` for `tabm` on California is about:
  - `0.450932`

### Comparison implication

- TabM is strong enough to include in the California comparison.
- TabM is not currently stronger than the validated local TabR run.
- So the right ordering is:
  1. lock TabR as champion candidate
  2. add TabM as the main dense-MLP baseline comparator
  3. align both to our California protocol
  4. only then run C1/C2

### C1/C2 framing update

- `C1` should become the aligned foundation comparison:
  - `TabR_on_our_split`
  - `TabM_on_our_split`
  - repo California references (`B1`, `G2`, `v3.5`, head-routing variants)
- `C2` should stay a decoder challenger:
  - change only the readout on top of the strongest aligned foundation
  - do not mix in encoder rewrites
