# Current Hypotheses

This file is generated from `docs/research/findings.jsonl`.
It is the current research surface for scaling decisions, not an append-only history.

Updated: 2026-03-23T19:43:50.595054+00:00

## How To Read This

- `cleared`: a hypothesis has strong supporting evidence and can be treated as established locally
- `partially_causal`: the factor is real and mattered, but does not fully explain the observed outcome
- `open`: still a live question for the next experiment
- `confounded`: result was informative, but interpretation depended on a known flaw or missing control
- `falsified`: do not keep spending on this version of the claim without a new mechanism

## Cleared

| Claim ID | Topic | Conclusion | Branch | Note |
|---|---|---|---|---|
| `afc-b-california-router-instability` | `afc_phase_b` | Because regression router training became non-finite on finite tokens. An explicit anchor-only fallback preserves prior quality and surfaces the failure mode with effective_defer_rate=0.0 and router_nonfinite_fallback=1. | `exp/afc-b-frozen-router` | `2026-03-23-afc-phase-b-california-router-instability.md` |

## Partially Causal

| Claim ID | Topic | Conclusion | Branch | Note |
|---|---|---|---|---|
| `afc-framework-contrastive-task-bank-sharpening` | `afc_framework` | Partially. The contrastive bank materially sharpened exact dataset reuse and lowered unseen-task neighborhood entropy, but the unseen segment neighbor ordering shifted rather than converging to an obviously cleaner taxonomy. The bank became more decisive faster than it became more semantically trustworthy. | `exp/afc-cross-dataset-lma` | `2026-03-23-afc-cross-dataset-lma-kickoff.md` |
| `afc-framework-feedback-updated-task-bank-loop` | `afc_framework` | Yes, partially. The task bank can now ingest champion/challenger outcome feedback and change future prior lookup, but the current live GraphDrone route still queries the bank as an anonymous task, so the benefit mainly propagates through similarity-weighted feedback rather than full exact task reuse. | `exp/afc-cross-dataset-lma` | `2026-03-23-afc-cross-dataset-lma-kickoff.md` |
| `afc-framework-live-task-exact-reuse` | `afc_framework` | Partially. Live GraphDrone now uses the real task key and exact-reuse blending on known datasets, which removes anonymous-query limitations and slightly improves the mean log-loss guardrail, but the routing policy still saturates defer and does not translate the stronger prior into a broad classification win. | `exp/afc-cross-dataset-lma` | `2026-03-23-afc-cross-dataset-lma-kickoff.md` |
| `afc-framework-live-task-prior-activation-surface` | `afc_framework` | Not on the current benchmark surface. The task-prior conditioning mechanism is now wired correctly, and a real config-loss bug was fixed, but the live classification contracts still did not expose a usable learned-router stage: multiclass remained static GeoPOE and the binary OOF portfolios collapsed to anchor-only before routing. | `exp/afc-cross-dataset-lma` | `2026-03-23-afc-cross-dataset-lma-kickoff.md` |
| `afc-framework-metadata-guided-task-neighborhoods` | `afc_framework` | Yes, partially. Using neighborhood targets derived from task-context signatures recovered a more plausible unseen segment neighbor ordering than contrastive-only training while staying sharper than reconstruction-only training. This is the best tradeoff so far between confidence and semantic plausibility, and it does so without dataset-name hardcoding in the production logic. | `exp/afc-cross-dataset-lma` | `2026-03-23-afc-cross-dataset-lma-kickoff.md` |
| `afc-framework-normalized-task-neighborhoods` | `afc_framework` | Yes, partially. On the six-dataset normalized classification bank, the held-out similarity space no longer collapsed to one deterministic prototype in every case. Several datasets now distribute probability across a small neighborhood of plausible related tasks, which is much closer to the intended transfer behavior. | `exp/afc-cross-dataset-lma` | `2026-03-23-afc-cross-dataset-lma-kickoff.md` |
| `afc-framework-persistent-task-prototype-bank` | `afc_framework` | Yes for exact dataset reuse, and partially for unseen-task lookup. The persistent prototype bank returns strong exact matches for known datasets and meaningful soft neighbors for unseen segment, though the neighborhood semantics are not yet sharp enough to count as a clean task taxonomy. | `exp/afc-cross-dataset-lma` | `2026-03-23-afc-cross-dataset-lma-kickoff.md` |
| `afc-framework-task-conditioned-lma-prior` | `afc_framework` | Probably yes. The within-task token-bank splits show classification has much tighter cross-dataset geometry than regression, while the mixed slice showed task type dominates family. That makes task type a real design axis for the next prior rather than a detail to ignore. | `exp/afc-cross-dataset-lma` | `2026-03-23-afc-cross-dataset-lma-kickoff.md` |
| `afc-framework-task-prior-defer-regularization` | `afc_framework` | Yes, partially. Task-prior-aware defer regularization is the first live binary mechanism on this branch that both desaturates defer materially and clears the current champion/challenger guardrails on the credit_g+diabetes slice. Strong regularization overcorrects and hurts F1, while a milder setting preserves the binary guardrails. This supports the policy-saturation hypothesis without yet establishing broad classification generalization. | `exp/afc-cross-dataset-lma` | `2026-03-23-afc-cross-dataset-lma-kickoff.md` |
| `afc-b-reg-anchor-asymmetry` | `afc_phase_b` | Yes. The asymmetry was real and partly causal, but fixing it was not sufficient to make Phase B promotable. | `exp/afc-b-reg-anchor-exclusion` | `2026-03-23-afc-phase-b-anchor-exclusion.md` |

## Open

| Claim ID | Topic | Conclusion | Branch | Note |
|---|---|---|---|---|
| `afc-framework-cross-dataset-hyper-lma` | `afc_framework` | This is now a live framework hypothesis. The current Phase B results increasingly suggest single-dataset router fitting may be too weak, because improved local diagnostics can still come mainly from routing suppression rather than better allocation. | `exp/afc-b-residual-objective` | `2026-03-23-afc-cross-dataset-lma-hypothesis.md` |
| `afc-framework-cross-dataset-token-geometry` | `afc_framework` | Partially, but not in the simplest form. In the first mixed regression/classification slice, task type separated tokens more strongly than view family, so any future LMA prior should likely be hierarchical or task-conditioned rather than a single family-invariant prior. | `exp/afc-cross-dataset-lma` | `2026-03-23-afc-cross-dataset-lma-kickoff.md` |
| `afc-framework-task-context-heldout-generalization` | `afc_framework` | Not reliably yet. In the first leave-one-dataset-out reconstruction check, both encoder families failed badly on optdigits while generalizing reasonably on the other held-out datasets. This keeps the task-conditioned direction alive, but shows that representation mismatch is still a major bottleneck. | `exp/afc-cross-dataset-lma` | `2026-03-23-afc-cross-dataset-lma-kickoff.md` |
| `afc-framework-task-context-prototype-learnability` | `afc_framework` | Yes, on the current bootstrap task-context surface. Both the transformer and GRU prototypes fit the four-dataset classification context set, and the transformer converged to a lower final loss on the same examples. | `exp/afc-cross-dataset-lma` | `2026-03-23-afc-cross-dataset-lma-kickoff.md` |
| `afc-framework-task-similarity-transfer-space` | `afc_framework` | Not yet. On the first similarity-aware held-out check, both encoders mapped each unseen dataset to exactly one seen prototype with near-total confidence, which is stable but too collapsed to count as a nuanced task-neighborhood space. | `exp/afc-cross-dataset-lma` | `2026-03-23-afc-cross-dataset-lma-kickoff.md` |
| `afc-b-residual-usefulness-gap` | `afc_phase_b` | On cpu_act, yes. Both champion and challenger have positive best-available specialist advantage but negative attention-weighted specialist advantage; rotor narrows the gap slightly on validation but still worsens held-out RMSE. California remains unresolved due NaN usefulness diagnostics. | `exp/afc-b-frozen-router` | `2026-03-23-afc-phase-b-residual-usefulness.md` |
| `afc-b-rotor-mechanism` | `afc_phase_b` | Rotor improves token alignment, but the current routing and integration design still does not convert that into a stable benchmark win. | `exp/afc-b-cayley-rotor` | `2026-03-23-afc-phase-b-claim-first.md` |

## Confounded

| Claim ID | Topic | Conclusion | Branch | Note |
|---|---|---|---|---|
| `afc-b-joint-training-interference` | `afc_phase_b` | Probably not as the dominant blocker. Freezing the base router did not improve the rotor challenger over the corrected non-frozen run, but branch-local champion drift prevents a clean falsification. | `exp/afc-b-frozen-router` | `2026-03-23-afc-phase-b-frozen-router.md` |

## Falsified

| Claim ID | Topic | Conclusion | Branch | Note |
|---|---|---|---|---|
| `afc-b-residual-objective-gap-penalty` | `afc_phase_b` | For the current objective shape, no. Across lambda={0.02,0.05,0.10} on the quick regression contract, the penalty was active but did not produce a better held-out model; its main effect was modest defer suppression rather than a genuinely better specialist-allocation policy. | `exp/afc-b-residual-objective` | `2026-03-23-afc-phase-b-residual-objective.md` |

## Recent Findings

| Timestamp | Claim ID | Status | Summary |
|---|---|---|---|
| `2026-03-23T19:43:50.594637+00:00` | `afc-framework-task-prior-defer-regularization` | `partially_causal` | Once task-bank identity and exact reuse were live, the remaining blocker proved to be defer saturation rather than prior availability. A task-prior-aware defer penalty moved mean defer out of the ~0.999 regime, and a mild setting produced the first promoted binary slice on this branch. |
| `2026-03-23T19:27:38.270058+00:00` | `afc-framework-live-task-exact-reuse` | `partially_causal` | Persistent task identity and exact-reuse blending are now active on the live binary route. That clears the bank/query identity bottleneck, but the remaining limiter is the binary defer policy rather than task-prior availability. |
| `2026-03-23T19:14:46.203797+00:00` | `afc-framework-feedback-updated-task-bank-loop` | `partially_causal` | The first feedback-updated bank iteration changed task-prior lookup and slightly improved the mean log-loss guardrail on the binary live rerun, but F1 stayed flat and defer remained saturated. This establishes a real closed loop without yet solving the live routing policy. |
| `2026-03-23T18:38:39.263083+00:00` | `afc-framework-live-task-prior-activation-surface` | `partially_causal` | The first live classification prototype showed that the task-prior bank is ready but the current GraphDrone classification architecture does not reliably expose an activation surface for it. The limiting factor moved from prior quality to router-stage availability. |
| `2026-03-23T18:28:00.917927+00:00` | `afc-framework-metadata-guided-task-neighborhoods` | `partially_causal` | The metadata-guided task bank derived neighborhood targets from dataset signatures rather than named-task rules. It kept segment nearest to pendigits on unseen-task lookup, unlike the contrastive-only bank, while remaining sharper than the reconstruction-only bank. |
| `2026-03-23T18:21:55.430449+00:00` | `afc-framework-contrastive-task-bank-sharpening` | `partially_causal` | Adding supervised contrastive pressure on top of reconstruction tightened exact-match similarities to about 0.997-0.999 and reduced unseen-segment entropy, but did not yet produce a stable semantically clean task-family assignment. |
| `2026-03-23T18:16:28.734975+00:00` | `afc-framework-persistent-task-prototype-bank` | `partially_causal` | The new prototype bank persists encoder state, normalization, and task centroids. Known datasets now return exact-reuse hits with high similarity, and an unseen segment dataset attaches to a soft neighborhood centered on pendigits/diabetes/credit_g instead of collapsing to a single arbitrary prototype. |
| `2026-03-23T18:06:30.959673+00:00` | `afc-framework-normalized-task-neighborhoods` | `partially_causal` | Normalization and a larger classification bank materially improved the task-conditioned prior: the similarity space now shows soft local neighborhoods like credit_g~diabetes/segment and pendigits~segment/optdigits instead of pure one-prototype collapse. |
| `2026-03-23T17:32:05.358173+00:00` | `afc-framework-task-similarity-transfer-space` | `open` | The task-conditioned prior now looks more like a hard prototype assigner than a useful similar-task transfer space. The next bottleneck is representation collapse, so the next work should target normalization and metric learning rather than only encoder choice. |
| `2026-03-23T17:25:43.001231+00:00` | `afc-framework-task-context-heldout-generalization` | `open` | The task-conditioned prior is learnable but not yet robustly transferable: held-out reconstruction remains fragile and is dominated by the optdigits outlier, so the next work should target feature normalization and dataset mismatch before claiming encoder-level wins. |
| `2026-03-23T17:19:23.203285+00:00` | `afc-framework-task-context-prototype-learnability` | `open` | The classification-first task-context prototype is learnable. Both encoders reached 100% accuracy on the current bootstrap dataset-identification task, with the transformer showing better final loss, so transformer should be the default baseline for the next stricter generalization check. |
| `2026-03-23T17:11:56.304425+00:00` | `afc-framework-task-conditioned-lma-prior` | `partially_causal` | The cross-dataset token-bank results now support a task-conditioned or hierarchical LMA prior: classification geometry is much more coherent than regression, so a flat universal prior is less defensible than a regime-aware transformer/GRU-style conditioning layer. |

