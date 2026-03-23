# GraphDrone Evidence Grade

## Evidence-Grade

Use this label when:
- the benchmark contract is explicit and unchanged across sides
- champion and challenger use the same dataset/fold surface
- comparison artifacts exist
- run provenance exists
- the result is strong enough to support a branch decision

Typical examples:
- mini-full champion/challenger fold 0
- full 3-fold champion/challenger

## Diagnostic-Grade

Use this label when:
- the run is intentionally narrow
- the goal is mechanism understanding, failure analysis, or smoke validation
- the result should not be treated as promotion evidence on its own

Typical examples:
- quick gate
- one-dataset or two-dataset probe
- targeted binary subset used to inspect log-loss vs F1 behavior

## Non-Comparable

Use this label when:
- cache or version drift is unresolved
- datasets or folds differ
- provenance or granular outputs are missing
- branch-local champion drift makes the interpretation ambiguous

If a run is non-comparable, do not argue from it. Fix the contract first.
