# Current Hypotheses

This file is generated from `docs/research/findings.jsonl`.
It is the current research surface for scaling decisions, not an append-only history.

Updated: 2026-03-23T16:13:20.166698+00:00

## How To Read This

- `cleared`: a hypothesis has strong supporting evidence and can be treated as established locally
- `partially_causal`: the factor is real and mattered, but does not fully explain the observed outcome
- `open`: still a live question for the next experiment
- `confounded`: result was informative, but interpretation depended on a known flaw or missing control
- `falsified`: do not keep spending on this version of the claim without a new mechanism

## Cleared

| Claim ID | Topic | Conclusion | Branch | Note |
|---|---|---|---|---|
| `afc-b-california-router-instability` | `afc_phase_b` | Because regression router training became non-finite on finite tokens. An explicit anchor-only fallback preserves prior quality and surfaces the failure mode with effective_defer_rate=0.0 and router_nonfinite_fallback=1. | `exp/afc-b-frozen-router` | `2026-03-23-afc-phase-b-california-router-instability.md` |

## Partially Causal

| Claim ID | Topic | Conclusion | Branch | Note |
|---|---|---|---|---|
| `afc-b-reg-anchor-asymmetry` | `afc_phase_b` | Yes. The asymmetry was real and partly causal, but fixing it was not sufficient to make Phase B promotable. | `exp/afc-b-reg-anchor-exclusion` | `2026-03-23-afc-phase-b-anchor-exclusion.md` |

## Open

| Claim ID | Topic | Conclusion | Branch | Note |
|---|---|---|---|---|
| `afc-framework-cross-dataset-hyper-lma` | `afc_framework` | This is now a live framework hypothesis. The current Phase B results increasingly suggest single-dataset router fitting may be too weak, because improved local diagnostics can still come mainly from routing suppression rather than better allocation. | `exp/afc-b-residual-objective` | `2026-03-23-afc-cross-dataset-lma-hypothesis.md` |
| `afc-b-residual-objective-gap-penalty` | `afc_phase_b` | The direct residual-usefulness penalty is active and reduces the diagnosed gap, but on the quick regression contract it does so mainly by collapsing defer rather than learning a clearly better specialist-allocation policy. | `exp/afc-b-residual-objective` | `2026-03-23-afc-phase-b-residual-objective.md` |
| `afc-b-residual-usefulness-gap` | `afc_phase_b` | On cpu_act, yes. Both champion and challenger have positive best-available specialist advantage but negative attention-weighted specialist advantage; rotor narrows the gap slightly on validation but still worsens held-out RMSE. California remains unresolved due NaN usefulness diagnostics. | `exp/afc-b-frozen-router` | `2026-03-23-afc-phase-b-residual-usefulness.md` |
| `afc-b-rotor-mechanism` | `afc_phase_b` | Rotor improves token alignment, but the current routing and integration design still does not convert that into a stable benchmark win. | `exp/afc-b-cayley-rotor` | `2026-03-23-afc-phase-b-claim-first.md` |

## Confounded

| Claim ID | Topic | Conclusion | Branch | Note |
|---|---|---|---|---|
| `afc-b-joint-training-interference` | `afc_phase_b` | Probably not as the dominant blocker. Freezing the base router did not improve the rotor challenger over the corrected non-frozen run, but branch-local champion drift prevents a clean falsification. | `exp/afc-b-frozen-router` | `2026-03-23-afc-phase-b-frozen-router.md` |

## Recent Findings

| Timestamp | Claim ID | Status | Summary |
|---|---|---|---|
| `2026-03-23T16:13:20.166398+00:00` | `afc-framework-cross-dataset-hyper-lma` | `open` | Cross-dataset latent manifold alignment / hyper-router priors are now recorded as the next-scale AFC hypothesis: if local objective fixes keep improving internals without strong held-out wins, the routing prior should be learned across datasets rather than from tiny per-dataset splits. |
| `2026-03-23T16:13:08.832205+00:00` | `afc-b-residual-objective-gap-penalty` | `open` | On the diagnostic quick regression contract, the new objective slightly reduced the negative held-out RMSE and sharply reduced defer-weighted negative specialist value, but mostly by routing less; keep the claim open and do not advance to mini-full yet. |
| `2026-03-23T14:00:43.692285+00:00` | `afc-b-california-router-instability` | `cleared` | California was not a token NaN problem. It was non-finite learned attention after regression router training, combined with negative realized specialist value. The new explicit fallback makes that visible while preserving the previous task quality. |
| `2026-03-23T13:41:40.482544+00:00` | `afc-b-residual-usefulness-gap` | `open` | The new residual-usefulness diagnostic showed that specialist value can exist while realized router value stays negative. Rotor slightly improved the validation usefulness score on cpu_act, but not enough to improve held-out quality, so allocation/objective mismatch remains the main live hypothesis. |
| `2026-03-23T13:25:48.287059+00:00` | `afc-b-joint-training-interference` | `confounded` | The frozen-base rotor ablation weakened the joint-training-interference hypothesis: it was flat to slightly worse than the corrected non-frozen rotor challenger, though same-branch champion drift means the result should be treated as informative rather than final. |
| `2026-03-23T04:08:57.496645+00:00` | `afc-b-reg-anchor-asymmetry` | `partially_causal` | Excluding anchor mass improved the rotor challenger much more than the champion path and largely removed the earlier strong negative result, but latency and remaining integration issues still block promotion. |
| `2026-03-23T04:08:57.463397+00:00` | `afc-b-rotor-mechanism` | `open` | Phase B claim-first review supported the mechanism but left translation open due to circuit design and metric coupling questions. |

