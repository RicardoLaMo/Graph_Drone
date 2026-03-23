# Current Hypotheses

This file is generated from `docs/research/findings.jsonl`.
It is the current research surface for scaling decisions, not an append-only history.

Updated: 2026-03-23T04:08:57.496850+00:00

## How To Read This

- `cleared`: a hypothesis has strong supporting evidence and can be treated as established locally
- `partially_causal`: the factor is real and mattered, but does not fully explain the observed outcome
- `open`: still a live question for the next experiment
- `confounded`: result was informative, but interpretation depended on a known flaw or missing control
- `falsified`: do not keep spending on this version of the claim without a new mechanism

## Partially Causal

| Claim ID | Topic | Conclusion | Branch | Note |
|---|---|---|---|---|
| `afc-b-reg-anchor-asymmetry` | `afc_phase_b` | Yes. The asymmetry was real and partly causal, but fixing it was not sufficient to make Phase B promotable. | `exp/afc-b-reg-anchor-exclusion` | `2026-03-23-afc-phase-b-anchor-exclusion.md` |

## Open

| Claim ID | Topic | Conclusion | Branch | Note |
|---|---|---|---|---|
| `afc-b-rotor-mechanism` | `afc_phase_b` | Rotor improves token alignment, but the current routing and integration design still does not convert that into a stable benchmark win. | `exp/afc-b-cayley-rotor` | `2026-03-23-afc-phase-b-claim-first.md` |

## Recent Findings

| Timestamp | Claim ID | Status | Summary |
|---|---|---|---|
| `2026-03-23T04:08:57.496645+00:00` | `afc-b-reg-anchor-asymmetry` | `partially_causal` | Excluding anchor mass improved the rotor challenger much more than the champion path and largely removed the earlier strong negative result, but latency and remaining integration issues still block promotion. |
| `2026-03-23T04:08:57.463397+00:00` | `afc-b-rotor-mechanism` | `open` | Phase B claim-first review supported the mechanism but left translation open due to circuit design and metric coupling questions. |

