# C2 Decoder Challenger Status

## Scope

`C2` is a decoder-only challenger on the aligned California split.

The intent is narrow:

- keep the TabM-style ensemble backbone
- compare a fixed mean-over-heads readout against a row-gated readout
- test whether decoder-side conditional contribution helps when the backbone is held fixed

## Current Evidence

### Earlier Repo Evidence

The prior head-gated decoder branch already showed that decoder-side changes can help our routed backbone:

- `HR_v2_diverse`: RMSE `0.4812`
- `HR_v4_headgated_diverse`: RMSE `0.4722`

So decoder credit assignment is a real issue in this repo.

### New C2 Full Result

On the aligned TabM full run:

- `C2a_TabM_mean_heads`: test RMSE `0.4578`
- `C2b_TabM_gated_heads`: test RMSE `0.4614`

The gate diagnostics were also weak:

- gate entropy `3.2041`
- top-1 gate mass `0.1067`

That is still diffuse for a 32-head gate. The gating distribution is not collapsing, but it is not becoming sharply selective either.

## Interpretation

The decoder hypothesis is still valid, but this particular `C2` design is too weak:

- it gates from head predictions alone
- it does not use stronger row-conditional signals such as retrieval quality, agreement structure, or uncertainty
- as a result, the gate learns little more than a soft average

The full result makes the point more strongly than the smoke result:

- the mean-head readout is better than the gated readout on both validation and test
- so a decoder change is not automatically an upgrade
- the gate needs informative conditioning features, not just a learnable softmax on head outputs

## Practical Conclusion

The lesson is not “decoder changes do not matter.”

The lesson is:

- decoder changes matter
- but naive post-hoc gating is not enough
- the decoder needs access to informative support-quality or routing-quality signals, otherwise it collapses toward uniform mixing

That is consistent with the TabR result: TabR’s advantage likely comes from a stronger target-conditioned support/value path, not just a fancier final MLP.
