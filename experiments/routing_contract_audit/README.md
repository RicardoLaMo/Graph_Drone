# Routing Contract Audit

> **Branch:** `feature/routing-contract-audit`
> **Isolated from:** all prior experiment folders — no prior files modified

## Purpose
Audit whether prior implementations truly understood routing vs merely doing post-hoc ensembling.

## Self-Check Alignment (confirmed before coding)
- **Routing:** `g_i → pi_i (view weights) + beta_i (mode gate [0,1])`, applied **before** final prediction
- **Not routing:** appending curvature to X, last-MLP with g, weighted logit ensemble, no explicit pi/beta
- **Model A:** post-hoc combiner, may use g but no explicit pi/beta
- **Model B:** explicit pi + beta + iso_rep + inter_rep + final_rep blend
- **beta→0** = isolation, **beta→1** = interaction (non-reversible)
- **First-pass scope:** per-row routing. Per-head extension path left open in code.
- **Geometry features:** inside `g` ONLY, not appended to X in Model B

## Run
```bash
source .venv/bin/activate
# 1. Routing contract tests (must pass before dataset run)
python experiments/routing_contract_audit/tests/test_routing_semantics.py -v

# 2. California Housing audit
python experiments/routing_contract_audit/scripts/run_audit.py --dataset california

# 3. MNIST audit
python experiments/routing_contract_audit/scripts/run_audit.py --dataset mnist
```

## Contract files
See `contract/` directory for the 3 authoritative spec files.

## Warning signs
- Model B loses to Model A on metrics AND shows no routing variance in pi/beta → deep failure
- pi is uniform for all rows → router learned nothing
- beta stays at 0.5 for all rows → mode gate learned nothing
