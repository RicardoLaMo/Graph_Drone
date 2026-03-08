# MQ-GoRA v4 Split-Track

Branch: `feature/mq-gora-v4-split-track`

## Purpose

MQ-GoRA v4 is a disciplined continuation of v3 under a validated bug-fix path.
It is split into two dataset tracks:

- California: regression-safe routing only
- MNIST: protected classification path that preserves the v3 G10 logic first

Geometry signals remain routing priors, not appended prediction features.

## What Changed From v3

- Explicit `beta` mode routing was added alongside `pi`.
- The v4 wrappers now restore the best checkpoint and validate teacher-query models with `z_anc`.
- California v4 removes raw label-context usage from the default path and only keeps normalised, train-masked label context in explicitly marked variants.
- California teacher-lite drops centroid loss.
- MNIST keeps an exact `G10_ref` path and only adds incremental diversity-pressure variants.

## Why Split Tracks

- California v3 rich-context variants underperformed G2 and appeared to overfit early.
- MNIST v3 genuinely benefited from the richer G8 → G9 → G10 ladder.
- A single shared training recipe would blur whether failures were regression-specific or truly architectural.

## Folder Layout

```text
experiments/mq_gora_v4/
  README.md
  shared/
    src/
    configs/
    reports/
    artifacts/
  california/
    configs/
    scripts/
    reports/
    figures/
    artifacts/
    logs/
  mnist/
    configs/
    scripts/
    reports/
    figures/
    artifacts/
    logs/
```

## Run Commands

Activate the repo environment first:

```bash
source .venv/bin/activate
```

Shared integrity:

```bash
python experiments/mq_gora_v4/shared/src/integrity_check.py
```

California smoke:

```bash
python experiments/mq_gora_v4/california/scripts/run_ca_v4.py --smoke
```

California default:

```bash
python experiments/mq_gora_v4/california/scripts/run_ca_v4.py
```

MNIST smoke:

```bash
python experiments/mq_gora_v4/mnist/scripts/run_mn_v4.py --smoke
```

MNIST default:

```bash
python experiments/mq_gora_v4/mnist/scripts/run_mn_v4.py
```

## Success Criteria

California success:

- at least one regression-safe v4 variant beats the failing v3 rich-context runs
- training stops later than the v3 collapse regime
- best variant moves materially back toward G2

MNIST success:

- v4 preserves or improves the saved G10 accuracy
- routing quality improves without material accuracy loss

Overall v4 success:

- integrity checks pass first
- routing behavior and predictive metrics both support the added complexity
- small gains are not oversold
