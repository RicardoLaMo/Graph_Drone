# GraphDrone Research Flow

1. Create a branch or worktree for one hypothesis.
2. Freeze git state and benchmark contract before coding.
3. Define the component claim and the local success signature.
4. Implement the minimum code change.
5. Run the smallest honest evaluation that can falsify the claim.
6. If the mechanism is mixed or negative, switch to `graphdrone-mechanism-diagnosis`.
7. Record:
- run lineage in `output/experiments/`
- note in `docs/`
- durable finding in `docs/research/`
8. Only then decide whether to broaden, narrow, or abandon the line.
