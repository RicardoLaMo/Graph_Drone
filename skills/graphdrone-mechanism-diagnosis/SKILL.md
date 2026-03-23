---
name: graphdrone-mechanism-diagnosis
description: Use when a GraphDrone component or mechanism looks locally promising but end-to-end metrics stay flat or negative, and you need to locate the break between component truth, routing/policy coupling, outcome translation, and architecture fit.
---

# GraphDrone Mechanism Diagnosis

This is the GraphDrone-specific realization of the methodology described in `doc/for_skill.md`: do not reject an idea from final RMSE, F1, or ELO alone. First determine whether the component worked, then whether the surrounding circuit used it correctly.

## Use This When

Use this skill when:
- claim reports say a mechanism is `supported` but the promotion decision is `hold`
- a new module was plugged in and the overall result worsened
- the right question is “why did the gain disappear?” rather than “who won?”
- you need a failure-chain narrative, not a benchmark scoreboard

## Required Inputs

Start from these GraphDrone artifacts when they exist:
- `comparison/claim_report.json`
- `comparison/promotion_decision.json`
- `comparison/paired_task_deltas.csv`
- `raw/.../run_ledger.json`
- `raw/.../run_events.jsonl`
- the current branch note in `docs/`

## Method

1. Write the change card.
- idea
- target component
- claimed bottleneck
- claimed mechanism
- expected local signature
- expected global signature
- likely failure modes

2. Separate four layers.
- `component truth`: did the intended signal move?
- `circuit/policy coupling`: did the router or integration logic change behavior in the expected direction?
- `outcome translation`: which metric family moved and which did not?
- `reproducibility control`: could the result be explained by seed drift, cache drift, or unstable training?

3. Prefer GraphDrone-native evidence before new instrumentation.
- alignment questions: `alignment_cosine_gain`
- routing allocation questions: `mean_specialist_mass`, `mean_anchor_attention_weight`, `effective_defer_rate`
- regression residual-usefulness questions:
  - `validation_best_specialist_advantage_score`
  - `validation_weighted_specialist_advantage_score`
  - `validation_defer_weighted_specialist_advantage_score`
  - `validation_top_specialist_advantage_score`
- instability questions: `router_nonfinite_fallback`

4. Force a failure bucket.
- Read `references/failure-buckets.md`.
- Pick one primary bucket and one secondary candidate.
- Do not stop at “method did not work.”

5. End with the minimum next discriminating check.
- not a giant rerun
- one change or one analysis that could falsify the current interpretation

## GraphDrone-Specific Heuristics

- If local mechanism gain exists but specialist mass or defer behavior moves the wrong way, suspect circuit coupling before rejecting the math.
- If log-loss improves but F1 does not, treat calibration/threshold mismatch as a real possibility.
- If diagnostics are `NaN` or routing goes non-finite on finite tokens, classify numerical stability before interpreting quality metrics.
- If a fix helps the challenger much more than the champion, treat the asymmetry as causal signal even if the branch still does not promote.

## Output Contract

Always answer:
- What claim was under test?
- Did the component activate?
- Did GraphDrone use it correctly?
- Which metric family moved, and which did not?
- What is the strongest current failure-chain explanation?
- What is the smallest next check that would falsify that explanation?

## Resources

- `references/failure-buckets.md`: GraphDrone-oriented classification for local-win/global-loss cases
- `references/question-ladder.md`: inside-out diagnostic prompts tuned for GraphDrone routing research
