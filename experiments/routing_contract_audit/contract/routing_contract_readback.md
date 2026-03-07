# Development-Agent Readback

Below is what a careful development agent should do after reading `routing_contract.md` and `test_routing_semantics.py`.

## What the agent should understand immediately

1. **Routing is not post-hoc ensembling.**
   The contract explicitly says observer features must control:
   - view trust through `pi`
   - isolation vs interaction through `beta`
   before final prediction.

2. **Curvature is not a predictive feature in the main design.**
   It may appear only inside the observer vector `g`, or as a control baseline if explicitly labeled.

3. **The first-pass audit excludes rank/compression routing.**
   The agent should not add rank selection, compression switching, or other “helpful” creativity in the first pass.

4. **Model A and Model B must share per-view encoders.**
   The agent should preserve fairness and isolate the routing difference.

## What code the agent should write next

A compliant development agent should create:

- `experiments/routing_comprehension_audit/`
- shared view encoders
- `A_posthoc_combiner`
- `B_intended_router`
- routing behavior reporting
- dataset adapters for California and MNIST if requested

But before any dataset run, the agent should execute:

```bash
python test_routing_semantics.py
```

and verify all tests pass.

## How the agent should structure Model A

Model A should:
- take per-view representations
- optionally concatenate observer vector `g`
- combine views in a standard learned post-hoc way
- not define explicit `pi` and `beta`

If the agent gives Model A explicit `pi` and `beta`, it has violated the contract.

## How the agent should structure Model B

Model B should:
- compute `pi` from `g`
- compute `beta` from `g`
- form `iso_rep`
- form `inter_rep`
- blend them into `final_rep`
- predict from `final_rep`

If the agent computes `pi` or `beta` but never uses them in representation combination, it has violated the contract.

## What the report should include

A compliant agent should produce a report that explicitly states:

- what routing means
- what `beta` means
- how Model A differs from Model B
- mean `pi` by view
- top-1 view frequency
- routing entropy
- mean `beta`
- regime-wise routing summaries

If those are missing, the agent has not fully executed the contract.

## Typical failure modes the agent should avoid

1. Appending curvature as a feature and calling it routing
2. Treating weighted view averaging as intended routing
3. Reversing `beta` semantics
4. Changing the shared per-view encoders between A and B
5. Adding extra routing ideas not requested in first pass
6. Skipping synthetic tests and going straight to dataset metrics

## Bottom-line interpretation

A careful development agent should treat this contract as executable specification, not prose inspiration.

The correct execution order is:

1. implement exact interfaces
2. pass synthetic routing tests
3. build dataset audit on top of that
4. report both predictive and routing-behavior evidence
