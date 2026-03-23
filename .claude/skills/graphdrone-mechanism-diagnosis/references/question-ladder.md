# GraphDrone Mechanism Question Ladder

## Component Truth

- What exact field in `claim_report.json` says the component activated?
- Is the activation finite across tasks or only on a subset?
- Is the claimed local signal stronger than trivial noise?

## Circuit Coupling

- Did `mean_specialist_mass`, `mean_anchor_attention_weight`, or `effective_defer_rate` move?
- Was that movement directionally helpful?
- Are useful specialists available but under-allocated?

## Outcome Translation

- Which metrics improved: RMSE, MAE, R2, F1, log-loss, latency?
- Could calibration improve while threshold metrics degrade?
- Is the “loss” actually a metric-family mismatch?

## Reproducibility

- Is the benchmark contract identical between champion and challenger?
- Is router seed fixed?
- Did any fallback path fire?
- Are there `NaN` or non-finite indicators that invalidate a naïve interpretation?

## Next Check

- What is the smallest code or analysis change that could prove this interpretation wrong?
