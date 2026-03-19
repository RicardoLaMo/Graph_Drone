"""
Geometric Product-of-Experts (GeoPOE) blending for classification.

Rationale
---------
Linear blending of probability distributions — (1-w)*p_A + w*p_B — violates the
probability simplex only in degenerate cases, but it conflates uncertainty and
disagreement and does not respect the Bayesian update structure.

Product-of-Experts (PoE) blends in log-space:
    log q(y|x) = Σ_i w_i · log p_i(y|x)  then renormalize
which is equivalent to a geometric mean of probabilities (uniform weights) or a
confidence-weighted geometric mean (heterogeneous weights).

Properties:
- Output is always a valid probability distribution (renormalized after log-blend)
- High-confidence specialists naturally dominate (low-entropy → high weight)
- Works for any number of classes C ≥ 2 with no mode-switch
- No training required — weights are computed at inference from predictive entropy
- Conservative: when specialists disagree, entropy rises → weights equalize →
  result approaches uniform geometric mean (safe fallback)
"""
from __future__ import annotations
import numpy as np
import torch


def _entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy along last axis.  p: [..., C] → [...] nats."""
    safe = np.clip(p, 1e-9, 1.0)
    return -np.sum(safe * np.log(safe), axis=-1)


def geo_poe_blend(
    predictions: np.ndarray,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Geometric Product-of-Experts blend over an expert ensemble.

    Parameters
    ----------
    predictions : np.ndarray, shape [N, E, C]
        Class probability matrices for N samples, E experts, C classes.
    temperature : float
        Sharpening parameter for confidence weights.
        temperature > 1 → sharper selection (most confident expert wins harder)
        temperature = 1 → standard inverse-entropy weighting
        temperature < 1 → softer weights (closer to uniform geometric mean)

    Returns
    -------
    blend : np.ndarray, shape [N, C]
        Blended class probabilities on the probability simplex.
    """
    assert predictions.ndim == 3, f"Expected [N, E, C], got {predictions.shape}"
    N, E, C = predictions.shape

    # Clamp to avoid log(0)
    p = np.clip(predictions, 1e-9, 1.0)
    log_p = np.log(p)  # [N, E, C]

    # Confidence weights: w_i ∝ exp(−H_i / temperature)
    # Lower entropy (more confident) → higher weight
    H = _entropy(p)                                      # [N, E]
    log_w = -H / temperature                             # [N, E]
    log_w -= np.max(log_w, axis=-1, keepdims=True)       # numerical stability
    weights = np.exp(log_w)                              # [N, E]
    weights /= weights.sum(axis=-1, keepdims=True)       # normalize → [N, E]

    # Log-space weighted blend  →  [N, C]
    log_blend = np.einsum("ne,nec->nc", weights, log_p)

    # Renormalize to probability simplex
    log_blend -= np.max(log_blend, axis=-1, keepdims=True)
    blend = np.exp(log_blend)
    blend /= blend.sum(axis=-1, keepdims=True)

    return blend.astype(np.float32)


def anchor_geo_poe_blend(
    predictions: np.ndarray,
    anchor_idx: int,
    *,
    anchor_weight: float = 2.0,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    GeoPOE variant that gives additional weight to the anchor (FULL) expert.

    The anchor is a TabPFN trained on the full feature set and full data.
    Boosting its weight guards against specialist subspace views introducing
    noise on low-information feature subsets.

    Parameters
    ----------
    predictions : np.ndarray, shape [N, E, C]
    anchor_idx : int
        Index of the FULL anchor expert in dimension E.
    anchor_weight : float
        Multiplicative boost applied to the anchor's confidence weight before
        renormalization.  anchor_weight=1 → standard GeoPOE.
    temperature : float
        Passed through to confidence weighting.

    Returns
    -------
    blend : np.ndarray, shape [N, C]
    """
    assert predictions.ndim == 3, f"Expected [N, E, C], got {predictions.shape}"
    N, E, C = predictions.shape

    p = np.clip(predictions, 1e-9, 1.0)
    log_p = np.log(p)

    H = _entropy(p)                                      # [N, E]
    log_w = -H / temperature
    log_w -= np.max(log_w, axis=-1, keepdims=True)
    weights = np.exp(log_w)                              # [N, E]

    # Boost anchor
    weights[:, anchor_idx] *= anchor_weight

    weights /= weights.sum(axis=-1, keepdims=True)

    log_blend = np.einsum("ne,nec->nc", weights, log_p)
    log_blend -= np.max(log_blend, axis=-1, keepdims=True)
    blend = np.exp(log_blend)
    blend /= blend.sum(axis=-1, keepdims=True)

    return blend.astype(np.float32)


def learned_geo_poe_blend_torch(
    log_p: torch.Tensor,
    defer_probs: torch.Tensor,
    specialist_weights: torch.Tensor,
    anchor_idx: int,
) -> torch.Tensor:
    """
    Torch-native learned GeoPOE blend — used during router training (differentiable).

    The anchor appears explicitly in the (1-d) term and is excluded from the
    specialist weight distribution to avoid double-counting.  Specialist weights
    are renormalized over non-anchor experts only.

    Parameters
    ----------
    log_p : torch.Tensor [N, E, C]
        Log-probabilities for each expert.
    defer_probs : torch.Tensor [N, 1]
        Learned defer probability (0 = anchor-only, 1 = full specialist blend).
    specialist_weights : torch.Tensor [N, E]
        Cross-attention weights over all E experts (anchor slot is zeroed/renormed out).
    anchor_idx : int
        Index of the FULL anchor expert in dimension E.

    Returns
    -------
    log_q : torch.Tensor [N, C]
        Blended log-probabilities (unnormalized; pass through log_softmax before NLL).
    """
    N, E, C = log_p.shape
    log_p_anchor = log_p[:, anchor_idx, :]                              # [N, C]

    # Exclude anchor from specialist weight distribution
    non_anchor = torch.ones(E, dtype=torch.bool, device=log_p.device)
    non_anchor[anchor_idx] = False

    if non_anchor.sum() == 0:
        # Edge case: only the anchor expert — skip blending entirely
        return log_p_anchor

    w_spec = specialist_weights[:, non_anchor]                          # [N, E-1]
    w_spec = w_spec / (w_spec.sum(dim=-1, keepdim=True) + 1e-9)        # renorm over specialists
    log_p_spec = log_p[:, non_anchor, :]                                # [N, E-1, C]

    log_p_blend = torch.einsum("ne,nec->nc", w_spec, log_p_spec)       # [N, C]
    log_q = (1.0 - defer_probs) * log_p_anchor + defer_probs * log_p_blend  # [N, C]
    return log_q


def learned_geo_poe_blend(
    predictions: np.ndarray,
    defer_probs: torch.Tensor,
    specialist_weights: torch.Tensor,
    anchor_idx: int,
) -> np.ndarray:
    """
    Inference-time learned GeoPOE blend (numpy I/O, torch routing tensors).

    Parameters
    ----------
    predictions : np.ndarray [N, E, C]
        Class probability matrices for N samples, E experts, C classes.
    defer_probs : torch.Tensor [N, 1]
        Learned defer probability from router.
    specialist_weights : torch.Tensor [N, E]
        Cross-attention specialist weights from router.
    anchor_idx : int
        Index of the FULL anchor expert in dimension E.

    Returns
    -------
    blend : np.ndarray [N, C]
        Blended class probabilities on the probability simplex.
    """
    p = np.clip(predictions, 1e-9, 1.0)
    log_p_np = np.log(p)                                                # [N, E, C]
    log_p_t = torch.tensor(log_p_np, dtype=torch.float32)

    d = defer_probs.cpu().detach()                                      # [N, 1]
    w = specialist_weights.cpu().detach()                               # [N, E]

    log_q = learned_geo_poe_blend_torch(log_p_t, d, w, anchor_idx).numpy()  # [N, C]

    # Renormalize to valid probability simplex
    log_q -= np.max(log_q, axis=-1, keepdims=True)
    q = np.exp(log_q)
    q /= q.sum(axis=-1, keepdims=True)
    return q.astype(np.float32)
