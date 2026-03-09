# MQ-GoRA v4 Split-Track Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Finish MQ-GoRA v4 on the existing `feature/mq-gora-v4-split-track` branch with explicit split-track routing semantics, integrity evidence, and the required California/MNIST reports and artifacts.

**Architecture:** Extend the current v4 scaffold rather than replacing it. Add an explicit observer-driven `beta` gate to the shared model/reporting path, then wire shared integrity/report utilities and dataset-specific run scripts to emit the prompt-required diagnostics, gates, and final reports while preserving v3 references and output isolation.

**Tech Stack:** Python 3.12, PyTorch, NumPy, Pandas, scikit-learn, matplotlib/seaborn, existing `experiments/gora_tabular` source.

---

### Task 1: Add failing tests for missing shared routing semantics

**Files:**
- Create: `tests/mq_gora_v4/test_shared_routing.py`
- Modify: `experiments/mq_gora_v4/shared/src/row_transformer_v4.py`
- Modify: `experiments/mq_gora_v4/shared/src/eval_v4.py`

**Step 1: Write the failing test**

- Assert `MQGoraTransformerV4` returns explicit `beta` with values in `[0, 1]`.
- Assert routing outputs change when observer inputs change.
- Assert `beta=0` selects isolation and `beta=1` selects interaction in the combiner helper.

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/mq_gora_v4/test_shared_routing.py -q`

**Step 3: Write minimal implementation**

- Add an explicit mode-routing head or helper in the shared v4 model path.
- Return `beta` in prediction/diagnostic outputs.

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/mq_gora_v4/test_shared_routing.py -q`

**Step 5: Commit**

`git commit -m "feat: add explicit beta routing to mq-gora v4"`

### Task 2: Add failing tests for shared report/integrity requirements

**Files:**
- Create: `tests/mq_gora_v4/test_reports.py`
- Modify: `experiments/mq_gora_v4/shared/src/eval_v4.py`
- Modify: `experiments/mq_gora_v4/shared/src/integrity_check.py`

**Step 1: Write the failing test**

- Assert integrity report includes self-alignment text and reference reproduction section.
- Assert report utilities emit metrics/regime/routing CSVs and required figure files.
- Assert root-cause/gate summaries can be written from shared helpers.

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/mq_gora_v4/test_reports.py -q`

**Step 3: Write minimal implementation**

- Expand `eval_v4.py` with routing diagnostics, figure generation, and markdown helpers.
- Expand `integrity_check.py` with reference comparison outputs and compatibility tables.

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/mq_gora_v4/test_reports.py -q`

**Step 5: Commit**

`git commit -m "feat: add mq-gora v4 integrity and reporting helpers"`

### Task 3: Wire California split-track outputs

**Files:**
- Modify: `experiments/mq_gora_v4/california/scripts/run_ca_v4.py`
- Create: `experiments/mq_gora_v4/california/configs/default.yaml`

**Step 1: Write the failing test**

- Assert California runner metadata contains the required variants and produces root-cause/gates/final reports in isolated California paths.

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/mq_gora_v4/test_reports.py -q`

**Step 3: Write minimal implementation**

- Emit `metrics.csv`, `regime_metrics.csv`, `routing_stats.csv`.
- Emit `root_cause_audit.md`, `gates_report.md`, `final_report.md`.
- Generate the required figures and California ablation table content.

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/mq_gora_v4/test_reports.py -q`

**Step 5: Commit**

`git commit -m "feat: add california mq-gora v4 reporting track"`

### Task 4: Wire MNIST split-track outputs

**Files:**
- Modify: `experiments/mq_gora_v4/mnist/scripts/run_mn_v4.py`
- Create: `experiments/mq_gora_v4/mnist/configs/default.yaml`

**Step 1: Write the failing test**

- Assert MNIST runner metadata contains protected baseline + incremental variants and produces the required isolated reports/figures/artifacts.

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/mq_gora_v4/test_reports.py -q`

**Step 3: Write minimal implementation**

- Emit MNIST metrics/regime/routing stats plus root-cause/gates/final reports.
- Preserve G10-style path and alpha-gate analysis.

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/mq_gora_v4/test_reports.py -q`

**Step 5: Commit**

`git commit -m "feat: add mnist mq-gora v4 reporting track"`

### Task 5: Add README, configs, and verification runs

**Files:**
- Create: `experiments/mq_gora_v4/README.md`
- Create: `experiments/mq_gora_v4/shared/configs/default.yaml`
- Modify: `progress.md`
- Modify: `findings.md`

**Step 1: Write the failing test**

- Assert README mentions branch name, split-track rationale, run commands, and success criteria.

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/mq_gora_v4/test_reports.py -q`

**Step 3: Write minimal implementation**

- Add README and config stubs.
- Run integrity check and smoke runs.
- Write generated reports with real measured results.

**Step 4: Run verification to verify it passes**

Run:
- `source .venv/bin/activate && pytest tests/mq_gora_v4 -q`
- `source .venv/bin/activate && python experiments/mq_gora_v4/shared/src/integrity_check.py`
- `source .venv/bin/activate && python experiments/mq_gora_v4/california/scripts/run_ca_v4.py --smoke`
- `source .venv/bin/activate && python experiments/mq_gora_v4/mnist/scripts/run_mn_v4.py --smoke`

**Step 5: Commit**

`git commit -m "docs: finalize mq-gora v4 reports and commands"`
