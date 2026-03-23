# GraphDrone Failure Buckets

Use these buckets when a mechanism looks locally good but the branch does not translate.

## 1. Adapter Missing

The new component needs a bridge that the current architecture does not provide.

Examples:
- new token transform but no matching routing consumer
- new view descriptor but static downstream blend

## 2. Circuit Coupling Mismatch

The component activates, but the router or defer policy uses it in the wrong direction.

Examples:
- positive alignment gain with worse defer allocation
- useful specialists exist, but weighted specialist value stays negative

## 3. Objective Mismatch

The optimization target rewards a property that is not equivalent to end-task quality.

Examples:
- align-to-anchor objective erases residual specialist value
- calibration improves while threshold metric degrades

## 4. Foundation Mismatch

The old pipeline assumptions are incompatible with the new method.

Examples:
- legacy anchor blending contaminates a specialist-only path
- router assumes a Euclidean structure while the new module changes geometry

## 5. Numerical Or Contract Instability

The mechanism may be fine, but the implementation path becomes unstable or silently falls back.

Examples:
- non-finite router gradients on finite tokens
- cache drift, version drift, or branch drift

## 6. Data Regime Mismatch

The idea helps only on certain datasets or metric regimes.

Examples:
- helps log-loss but not F1
- helps large datasets but not the mini-full contract overall

## 7. Mechanism Illusion

The local win is real but not causally connected to the downstream metric.

Examples:
- alignment improves, but the improved geometry is irrelevant to predictive residuals

Choose one primary bucket. If several are plausible, rank them rather than averaging them into a vague answer.
