# P0 Multiseed Summary

Aligned California split is fixed at `split_seed=42`. Model and router randomness vary by `seed`.

| Seed | P0_FULL | P0_router | Gain |
|---|---:|---:|---:|
| 41 | 0.3982 | 0.3812 | +0.0170 |
| 42 | 0.3891 | 0.3773 | +0.0118 |
| 43 | 0.3922 | 0.3784 | +0.0138 |

- `P0_FULL` mean test RMSE: `0.3932 ± 0.0046`
- `P0_router` mean test RMSE: `0.3790 ± 0.0020`
- mean paired gain: `+0.0142`

Interpretation:
- The learned router beats the single global TabPFN expert on all three seeds.
- The gain is not an ensemble-only effect: `P0_uniform` and `P0_sigma2` are both much worse than `P0_router` in the seed-42 full report.
- This is still a meta-routing protocol, not a pure zero-shot PFN baseline. The router is trained on a train/holdout split carved from validation rows, then evaluated once on test.
