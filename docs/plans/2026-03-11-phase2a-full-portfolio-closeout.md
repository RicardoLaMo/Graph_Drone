# 2026-03-11 Phase II-A Full Portfolio Closeout

## Scope

Branch: `codex/graphdrone-fit-phase2a-portfolio`

Objective:
- evaluate the new `GraphDrone.fit()` path on the registered regression portfolio
- keep the public benchmark surface to:
  - `GraphDrone`
  - `TabPFN`
  - `TabR`
  - `TabM`
- measure whether the Phase II-A geometry-expert integration holds up beyond smoke mode

Reports root:
- `experiments/openml_regression_benchmark/reports_phase2a_full_portfolio`

## Completion

- full benchmark completed successfully
- total dataset-fold jobs: `27 / 27`
- datasets: `9`
- folds per dataset: `3`
- all public model rows were written for every dataset-fold

## Portfolio Leaderboard

| Dataset | GraphDrone | TabPFN | TabR | TabM | Best |
|---|---:|---:|---:|---:|---|
| `airfoil_self_noise` | `1.3275` | `1.2742` | `1.6993` | `1.4153` | `TabPFN` |
| `california_housing_openml` | `0.3943` | `0.4076` | `0.4053` | `0.4555` | `GraphDrone` |
| `concrete_compressive_strength` | `4.3203` | `4.3306` | `5.4709` | `5.1452` | `GraphDrone` |
| `diamonds` | `516.9677` | `512.7602` | `546.1741` | `532.0915` | `TabPFN` |
| `healthcare_insurance_expenses` | `4510.7174` | `4540.5938` | `4978.0528` | `4591.8505` | `GraphDrone` |
| `houses` | `0.1990` | `0.2037` | `0.2048` | `0.2290` | `GraphDrone` |
| `miami_housing` | `81762.6177` | `80132.0483` | `89187.9895` | `89375.8604` | `TabPFN` |
| `used_fiat_500` | `734.4574` | `732.0692` | `780.3606` | `767.5875` | `TabPFN` |
| `wine_quality` | `0.6515` | `0.6465` | `0.6598` | `0.6748` | `TabPFN` |

Win counts:
- `GraphDrone`: `4 / 9`
- `TabPFN`: `5 / 9`

## Main Read

What Phase II-A supports:
- the new package path is not a smoke artifact
- `GraphDrone.fit()` is portfolio-competitive
- `GraphDrone` beats `TabPFN` on:
  - `california_housing_openml`
  - `concrete_compressive_strength`
  - `healthcare_insurance_expenses`
  - `houses`
- `GraphDrone` beats both `TabR` and `TabM` on all `9 / 9` datasets in this portfolio

What Phase II-A does not support:
- a claim that `GraphDrone` is now best overall
- a claim that geometry experts alone solve the remaining gap

## Geometry Expert Read

Across all `27` folds:
- best geometry expert beat the anchor in `14 / 27` folds

Per dataset:
- `concrete_compressive_strength`: `3 / 3`
- `healthcare_insurance_expenses`: `3 / 3`
- `miami_housing`: `3 / 3`
- `used_fiat_500`: `2 / 3`
- `california_housing_openml`: `1 / 3`
- `diamonds`: `1 / 3`
- `airfoil_self_noise`: `1 / 3`
- `houses`: `0 / 3`
- `wine_quality`: `0 / 3`

Interpretation:
- geometry specialists are real and useful on multiple datasets
- but they do not yet consistently translate into a public `GraphDrone` win
- the remaining bottleneck is integration quality, not merely expert existence

## Decision

Phase II-A should be treated as a successful architecture checkpoint:
- the benchmark surface is cleaned up
- the `GraphDrone.fit()` package works across the registered portfolio
- geometry experts are integrated into the actual model path
- the remaining work is now a Phase II-B problem:
  - improve specialist harvesting and routing quality
  - not re-litigate whether the package-level architecture is viable
