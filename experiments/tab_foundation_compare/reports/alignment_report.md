# M5 Alignment Report

## Purpose

Move the foundation models onto the repo California protocol before any challenger swap.

## Repo California Protocol

- Dataset: California Housing
- Split: `70/15/15`, seed `42`
- Feature edit: `log1p` on columns `2` and `4`
- Evaluation: RMSE on the repo-aligned split

## Aligned Results Snapshot

- `TabR_on_our_split`
  - test RMSE `0.3829`
  - val RMSE `0.4154`
  - best epoch `62`
- `TabM_on_our_split`
  - test RMSE `0.4290`
  - val RMSE `0.4538`
  - best step `12426`

## Interpretation

- Alignment did not weaken the champion story. It strengthened it.
- `TabR` improved relative to its earlier local fixed run (`0.3949 -> 0.3829`).
- `TabM` also improved relative to its upstream shipped seed-0 reference (`0.4418 -> 0.4290`).

## What This Means

The repo California protocol is not harsher than the upstream paths these models were first anchored on. If anything, it is somewhat easier for both foundations.

That matters for fairness:

- future challenger deltas must be compared on the aligned protocol
- upstream paper numbers and repo numbers should not be mixed in one table without labeling the protocol
