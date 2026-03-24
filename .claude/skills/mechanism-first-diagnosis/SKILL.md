---
name: mechanism-first-diagnosis
scope: global
description: Use when a new method shows a local or component-level gain but the end-to-end result is flat or negative, and the real task is to explain the break between mechanism truth, integration, and final outcome.
---

# Mechanism-First Diagnosis

Its job is not to declare "the method failed" from the final metric alone.
Its job is to locate where the gain disappeared.

## Non-Negotiable Rules

- Never reject an idea from final benchmark scores alone.
- Never skip component-level evidence.
- If local gain exists but global gain does not, explain the failure chain.
- If the failure chain is still unclear, do not collapse the result into a simple reject.

## Required Inputs

Start from whatever component-level and system-level evidence exists:
- component diagnostic signals (before and after the change)
- policy or routing allocation records
- headline benchmark results (claim report, promotion decision, or equivalent)
- branch note or experiment log
- run provenance (seeds, versions, contract)

## Workflow

1. **Write the change card.**
   - what changed
   - what bottleneck it targets
   - what local signature should improve
   - what end-to-end signature should improve
   - likely failure modes

2. **Separate four layers.**
   - `component truth`: did the intended signal move?
   - `circuit / policy coupling`: did the integration layer use it in the right direction?
   - `outcome translation`: which metric family moved and which did not?
   - `reproducibility control`: could seed drift, version drift, or contract drift explain the result?

3. **Use available evidence before adding new instrumentation.**
   - Check component diagnostics first.
   - Only add new probes if the existing evidence is genuinely ambiguous.

4. **Force a primary failure bucket.**
   - Read `references/failure-buckets.md`.
   - Pick one primary bucket and one secondary candidate.
   - Do not stop at "method did not work."

5. **End with the minimum next discriminating check.**
   - Not a giant rerun.
   - One change or one analysis that could falsify the current interpretation.

## Output Contract

Always answer:
- What claim was under test?
- Did the component activate as intended?
- Did the surrounding system use it correctly?
- Which metric family moved, and which did not?
- What is the strongest current failure-chain explanation?
- What is the smallest next check that would falsify that explanation?

## Heuristics

- If local component gain exists but downstream allocation or policy moves the wrong way, suspect circuit coupling before rejecting the math.
- If calibration metric improves but threshold metric degrades, treat objective mismatch as likely.
- If diagnostics are undefined or routing falls back silently, classify numerical stability before interpreting quality metrics.
- If a fix helps the weaker side more than the stronger side, treat the asymmetry as causal signal even if the branch does not promote.

## Adaptation Notes

Project-specific versions should bind this workflow to concrete artifacts such as claim reports, routing diagnostics, or calibration reports.

Read `references/failure-buckets.md` and `references/question-ladder.md`.
