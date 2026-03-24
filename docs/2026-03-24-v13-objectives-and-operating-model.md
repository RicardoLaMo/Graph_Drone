# GraphDrone v1.3 Objectives And Operating Model

Date: 2026-03-24

Branch:
- `exp/v13-reg-afc-revisit`

Context:
- champion = `v1.2`
- challengers = `v1.3` research lanes

## Core Objective

`v1.3` is not a leaderboard-only release train.
It is a research-model program with two linked goals:

1. improve GraphDrone beyond the `v1.2` champion in a causally defensible way
2. build a stronger architecture and research memory so future scaling does not restart from scratch

That means the decision rule is:
- mechanism-first
- benchmark-second

`v1.3` only counts as real progress if we can explain:
- what local mechanism changed
- how it changed routing or allocation behavior
- why that did or did not improve held-out prediction quality

## Champion / Challenger Contract

Champion:
- current accepted GraphDrone `v1.2`

Challengers:
- `v1.3` branches that test one mechanism at a time

Required contract:
- identical dataset / fold / cache boundaries
- explicit preset and version isolation
- branch-backed notes and artifacts
- durable finding written to `docs/research/findings.jsonl`

This avoids drifting into:
- benchmark-only story telling
- unfair reruns
- mechanism claims without evidence

## Required Skills

The `v1.3` program should explicitly use these skills together:

### `graphdrone-benchmark-governance`

Use for:
- champion/challenger fairness
- dataset/fold/version/cache discipline
- deciding when a run is exploratory versus promotable evidence

Program rule:
- no benchmark result is accepted without a clear contract

### `graphdrone-mechanism-diagnosis`

Use for:
- separating component truth from policy coupling and outcome translation
- explaining why a local win does not become a global win
- deciding whether failure is due to wiring, circuit design, or objective mismatch

Program rule:
- every strong claim needs a mechanism path, not only a metric delta

### `graphdrone-research-memory`

Use for:
- turning runs into durable findings
- keeping open, cleared, partially causal, confounded, and falsified claims visible
- making handoffs cumulative rather than conversational

Program rule:
- every serious lane writes to research memory

### `graphdrone-research-ops`

Use for:
- branch/worktree hygiene
- traceable notes
- provenance artifacts
- handoff-ready research packaging

Program rule:
- every meaningful experiment is branch-backed and note-backed

## v1.3 Phase Map

The current `v1.3` program should be read as:

### Phase 1: Regression Stability

Goal:
- make route-state and fallback behavior fully diagnosable

Current status:
- accomplished enough to use as a stable baseline

What is established:
- regression route state is now separated into clean routed, legitimacy-gate early exit, and router fallback
- fallback stage and reason are surfaced in artifacts
- regression task-prior plumbing already exists in code, so LMA is blocked by architecture readiness rather than missing hooks

Reference:
- `docs/2026-03-23-v13-reg-stability.md`

### Phase 2: Regression Usefulness Routing

Goal:
- convert positive available specialist value into positive realized specialist value

Current status:
- accomplished as a diagnosis phase, not as a solved routing design

What is established:
- direct allocation pressure can change routing behavior
- blunt global allocation rewards are not a general fix
- selective scalar penalties are safer, but still not the right final regression circuit

Reference:
- `docs/2026-03-23-v13-reg-usefulness-routing.md`

### Phase 3B: Classification LMA / Task Prior Learning

Goal:
- learn task-conditioned priors and persistent task memory so GraphDrone can reuse prior structure on the same or similar datasets

Current status:
- accomplished as a framework and architecture phase

What is established:
- task-conditioned LMA is viable, especially classification-first
- persistent task prototype banks and exact task reuse are real
- metadata-guided neighborhoods are better than naive collapsed retrieval
- live task-prior routing can be activated
- the remaining blocker is policy/routing behavior, not the existence of the prior itself

Reference:
- `docs/2026-03-23-afc-cross-dataset-lma-kickoff.md`
- `docs/2026-03-23-afc-research-handoff.md`

## What v1.3 Has Accomplished So Far

Across phases 1, 2, and 3B, `v1.3` has already accomplished four important things:

1. it turned GraphDrone research into a reproducible operating system
2. it exposed real routing-path and fallback-path structure instead of opaque benchmark outcomes
3. it proved that task priors and LMA-style memory are real architectural substrates, not only ideas
4. it narrowed the current blockers to routing-policy translation rather than missing infrastructure

So the program is no longer asking:
- can we add task priors?
- can we run champion/challenger?

It is now asking:
- how do we make routing use those priors correctly?
- how do we turn geometric or task-level signal into better specialist allocation?

## Current Objective By Track

Regression:
- keep `v1.2` as champion
- improve routing usefulness before expanding to regression meta-priors
- use AFC only when it can improve allocation translation, not alignment alone

Classification:
- keep the LMA/task-prior line alive
- treat persistent task memory as established substrate
- focus next on routing policy and operating-point/ranking behavior

Shared architecture objective:
- GraphDrone should learn from prior tasks and reuse that structure when the same or similar dataset appears again
- the latent manifold should store reusable routing knowledge, not only descriptor similarity

## What We Should Not Do

Do not:
- collapse back to TabArena-style winner/loser thinking
- treat a tiny benchmark win as success without mechanism explanation
- restart regression with broad LMA before local routing is ready
- claim AFC success from cosine gain alone

## Next-Step Discipline

Every next `v1.3` experiment should answer all four:

1. what mechanism is being tested?
2. what branch and artifact paths prove it?
3. what changed in route/allocation behavior?
4. should the next team keep spending on it?

That is the working definition of `v1.3` progress.
