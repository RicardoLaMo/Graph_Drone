"""
meta_learner.py — v3 meta-learning components for MQ-GoRA.

Four modules that address the four root causes identified in v2:

ViewSpecificEmbedder   (RC2: shared embedder → view-blind values)
  M separate Linear+LayerNorm projections, one per view.
  Used in STAGE A of the student forward pass.

LabelContextEncoder    (RC4: no label context in routing)
  Encodes per-view neighbour label statistics into:
    - label_ctx_vec [B, d_z]: router-input vector summarising all-view label info
    - lbl_delta     [B, K, M, d_model]: per-slot label value augmentation (in V_m)
    - lbl_weight    [B, K, M]: soft attention mask (down-weights label-noisy positions)
  Zero-input-safe: W_lbl_v zero-init → no effect at inference when lbl_nei=None.

ManifoldReader         (RC3: meta-learner blind to neighbourhood content)
  Aggregates each view's neighbourhood into ctx_vec [B, d_model].
  Two modes controlled by `use_query_z`:
    False (G7/G8): simple edge-weight-masked avg-pool of h_nei_m per view
    True  (G9/G10): z_anc-guided cross-attention — teacher z acts as query,
                    asking "from my manifold position, what matters in each view?"
  Outputs ctx_vec = Σ_m (learned_view_weight_m · ctx_m).

AlphaGate              (G10 only: α-gated local-vs-transformer prediction fusion)
  pred_local = W_local(ctx_vec)
  alpha      = σ( W_alpha( [z_anc ; ctx_vec] ) )
  pred_final = (1-α) · pred_base + α · pred_local
  Auxiliary loss: L_alpha = MSE(alpha, agree_score_batch)
  → High agree → α small (trust multi-view transformer synthesis)
  → Low agree  → α large (trust view-specific local reading)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

EPS = 1e-6


# ─── ViewSpecificEmbedder ─────────────────────────────────────────────────────

class ViewSpecificEmbedder(nn.Module):
    """
    M separate Linear(d_x → d_model) + LayerNorm projections.

    forward(x_nei_per_view: list of [B, K, d_x]) → list of [B, K, d_model]

    One embedder per view; structurally prevents head collapse by ensuring
    value vectors are view-discriminative from the start.
    """

    def __init__(self, n_views: int, d_x: int, d_model: int):
        super().__init__()
        self.n_views = n_views
        self.embedders = nn.ModuleList([
            nn.Sequential(nn.Linear(d_x, d_model), nn.LayerNorm(d_model))
            for _ in range(n_views)
        ])

    def forward(self, x_nei_per_view: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        x_nei_per_view: list of M tensors, each [B, K_m, d_x]
        Returns: list of M tensors, each [B, K_m, d_model]
        """
        return [emb(x_nei_per_view[m]) for m, emb in enumerate(self.embedders)]

    def embed_anchor(self, x_anc: torch.Tensor) -> torch.Tensor:
        """
        Embed anchor using view 0's embedder (shared convention).
        Alternatively use this for a separate anchor projection.
        x_anc: [B, d_x] → [B, d_model]
        """
        return self.embedders[0](x_anc)


# ─── LabelContextEncoder ─────────────────────────────────────────────────────

class LabelContextEncoder(nn.Module):
    """
    Encodes per-view neighbour label statistics for dual use:
      (a) Router input: label_ctx_vec [B, d_z]
      (b) Value augmentation: lbl_delta [B, K, M, d_model]

    All weights that touch lbl_nei are zero-initialised so that at inference
    (lbl_nei=None → zeros), the output is a stable learned bias, not noise.

    label_ctx_m_i = Σ_k (ew[k,m] · mask[k,m] · y_nei[k,m]) / Σ_k(ew·mask + ε)

    For classification:
      y_nei should be normalised class probabilities [B, K, M, C] OR
      one-hot labels [B, K, M] (integer class index).
    """

    def __init__(self, n_views: int, d_model: int, d_z: int,
                 n_out: int = 1):
        """
        n_out: 1 for regression, n_classes for classification.
        """
        super().__init__()
        self.n_views = n_views
        self.n_out = n_out

        # Value augmentation: lbl_delta_m = W_lbl_v · y_nei_m  [B, K, d_model]
        # One per view, zero-init
        self.W_lbl_v = nn.ModuleList([
            nn.Linear(n_out, d_model, bias=False) for _ in range(n_views)
        ])
        for lv in self.W_lbl_v:
            nn.init.zeros_(lv.weight)

        # Soft attention mask weight per view: scalar correction to attention logit
        # w_mask_m · (y_nei_m - label_ctx_m)² → penalise label-noisy positions
        self.log_w_mask = nn.Parameter(torch.zeros(n_views))  # exp → ≥0

        # Router context vector: MLP over all-view label centroids
        self.mlp_label = nn.Sequential(
            nn.Linear(n_views * n_out, d_z), nn.GELU(),
            nn.Linear(d_z, d_z),
        )
        # Zero-init first layer so output starts as a learned bias (stable inference)
        nn.init.zeros_(self.mlp_label[0].weight)

    def forward(
        self,
        lbl_nei: Optional[torch.Tensor],   # [B, K, M] (reg) or [B, K, M, C] (clf)
        ew_anc: torch.Tensor,              # [B, K, M]
        view_mask: torch.Tensor,           # [B, K, M]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          label_ctx_vec: [B, d_z]           — for router input
          lbl_delta:     [B, K, M, d_model] — for value augmentation in ManifoldReader
          lbl_weight:    [B, K, M]          — soft attention mask (additive logit correction)
        """
        B, K, M = ew_anc.shape
        device = ew_anc.device

        if lbl_nei is None:
            # Inference: zeros everywhere (W_lbl_v zero-init → lbl_delta=0; mlp bias gives ctx_vec)
            lbl_nei = torch.zeros(B, K, M, self.n_out, device=device)
        elif lbl_nei.dim() == 3:
            # Regression: [B, K, M] → [B, K, M, 1]
            lbl_nei = lbl_nei.unsqueeze(-1)
        # lbl_nei: [B, K, M, n_out]

        # Per-view label centroid (edge-weight-masked mean)
        wm = ew_anc * view_mask                                  # [B, K, M]  valid weights
        w_sum = wm.sum(dim=1, keepdim=True).clamp(min=EPS)      # [B, 1, M]
        # label_ctx_m: [B, M, n_out]
        label_ctx_m = (wm.unsqueeze(-1) * lbl_nei).sum(1) / w_sum.squeeze(1).unsqueeze(-1).clamp(min=EPS)

        # Router input vector
        flat_ctx = label_ctx_m.view(B, M * self.n_out)
        label_ctx_vec = self.mlp_label(flat_ctx)                 # [B, d_z]

        # Value augmentation: lbl_delta_m = W_lbl_v_m(y_nei_m)  [B, K, d_model]
        lbl_delta_list = []
        for vi in range(M):
            delta_v = self.W_lbl_v[vi](lbl_nei[:, :, vi, :])   # [B, K, d_model]
            lbl_delta_list.append(delta_v)
        lbl_delta = torch.stack(lbl_delta_list, dim=2)          # [B, K, M, d_model]

        # Soft attention mask: down-weight positions where y_nei deviates from centroid
        # mask_logit_m = -w_mask * ||y_nei - label_ctx_m||²
        w_mask = torch.exp(self.log_w_mask)                     # [M] positive
        # deviation: [B, K, M, n_out] - [B, 1, M, n_out]
        deviation = lbl_nei - label_ctx_m.unsqueeze(1)
        deviation_sq = (deviation ** 2).sum(-1)                 # [B, K, M]
        # negative because higher deviation → lower logit (soft-mask away noisy positions)
        lbl_weight = -w_mask.view(1, 1, M) * deviation_sq       # [B, K, M]

        return label_ctx_vec, lbl_delta, lbl_weight


# ─── ManifoldReader ───────────────────────────────────────────────────────────

class ManifoldReader(nn.Module):
    """
    Aggregates each view's neighbourhood into ctx_vec [B, d_model].

    Two modes:
      use_query_z=False (G7/G8): edge-weight avg-pool (baseline, no teacher)
      use_query_z=True  (G9/G10): z_anc cross-attention query (manifold-guided read)

    Cross-attention (G9/G10):
      Q = W_Q(z_anc)                        [B, d_model]
      For view m:
        K_m = W_K_m(h_nei_m)               [B, K, d_model]
        V_m = W_V_m(h_nei_m) + lbl_delta_m [B, K, d_model]
        score_m = Q·K_m^T/√d + log(ew_m+ε) + lbl_weight_m, masked
        attn_m = softmax(score_m)            [B, K]
        ctx_m = attn_m · V_m                [B, d_model]

      ctx_vec = Σ_m view_weight_m · ctx_m   [B, d_model]
      view_weight_m = softmax(log_vw)[m]    scalar per view (learned)
    """

    def __init__(self, n_views: int, d_z: int, d_model: int,
                 use_query_z: bool = True, dropout: float = 0.1):
        super().__init__()
        self.n_views = n_views
        self.d_model = d_model
        self.use_query_z = use_query_z

        if use_query_z:
            self.W_Q = nn.Linear(d_z, d_model, bias=False)

        # Per-view key and value projections
        self.W_K = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(n_views)])
        self.W_V = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(n_views)])

        # Learned view importance weights (shared scalar per view, softmax normalised)
        self.log_view_weight = nn.Parameter(torch.zeros(n_views))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_nei_list: List[torch.Tensor],    # M x [B, K, d_model]
        ew_anc: torch.Tensor,              # [B, K, M]
        view_mask: torch.Tensor,           # [B, K, M]
        z_anc: Optional[torch.Tensor] = None,   # [B, d_z]  — None in avg-pool mode
        lbl_delta: Optional[torch.Tensor] = None,   # [B, K, M, d_model]  — or None
        lbl_weight: Optional[torch.Tensor] = None,  # [B, K, M]  — or None
    ) -> torch.Tensor:
        """
        Returns ctx_vec [B, d_model].
        """
        B, K, M = ew_anc.shape
        d = self.d_model

        if self.use_query_z and z_anc is not None:
            Q = self.W_Q(z_anc)              # [B, d_model]
        else:
            Q = None

        view_weights = torch.softmax(self.log_view_weight, dim=0)   # [M]
        ctx_parts = []

        for vi in range(M):
            h_m = h_nei_list[vi]                      # [B, K, d_model]

            # Value augmentation with label delta
            V_m = self.W_V[vi](h_m)                  # [B, K, d_model]
            if lbl_delta is not None:
                V_m = V_m + lbl_delta[:, :, vi, :]   # [B, K, d_model]

            if Q is not None:
                # Cross-attention: Q [B, d] · K_m [B, K, d]^T → [B, K]
                K_m = self.W_K[vi](h_m)              # [B, K, d_model]
                scores = (Q.unsqueeze(1) * K_m).sum(-1) / math.sqrt(d)   # [B, K]

                # Geometry bias: log(ew_m + eps)
                ew_m = ew_anc[:, :, vi]              # [B, K]
                scores = scores + torch.log(ew_m + EPS)

                # Label soft-mask
                if lbl_weight is not None:
                    scores = scores + lbl_weight[:, :, vi]  # [B, K]

                # Mask absent neighbours (view_mask == 0)
                absent = (view_mask[:, :, vi] == 0)
                scores = scores.masked_fill(absent, -1e9)

                attn_m = self.dropout(torch.softmax(scores, dim=-1))   # [B, K]
            else:
                # Avg-pool mode: edge-weight-masked mean
                ew_m = ew_anc[:, :, vi] * view_mask[:, :, vi]          # [B, K]
                w_sum = ew_m.sum(-1, keepdim=True).clamp(min=EPS)
                attn_m = ew_m / w_sum                                   # [B, K]

            ctx_m = (attn_m.unsqueeze(-1) * V_m).sum(1)   # [B, d_model]
            ctx_parts.append(view_weights[vi] * ctx_m)

        ctx_vec = torch.stack(ctx_parts, dim=0).sum(0)     # [B, d_model]
        return ctx_vec


# ─── AlphaGate ────────────────────────────────────────────────────────────────

class AlphaGate(nn.Module):
    """
    α-gated prediction fusion (G10 only).

    pred_local = W_local(ctx_vec)                         [B, out_dim]
    alpha      = σ( W_alpha( concat[z_anc, ctx_vec] ) )  [B, 1]
    pred_final = (1-α) · pred_base + α · pred_local

    Auxiliary loss:
      L_alpha = MSE(alpha.squeeze(-1), agree_score_batch)
      → High agreement → α small (trust transformer's multi-view synthesis)
      → Low agreement  → α large (trust local view-specific reading)

    W_alpha near-zero init ensures α≈0.5 at epoch 0 (safe start).
    """

    def __init__(self, d_z: int, d_model: int, out_dim: int):
        super().__init__()
        self.W_local = nn.Linear(d_model, out_dim)
        self.W_alpha = nn.Linear(d_z + d_model, 1)
        # Near-zero init: α≈0.5 at start, network learns when to trust local
        nn.init.normal_(self.W_alpha.weight, std=0.01)
        nn.init.zeros_(self.W_alpha.bias)

    def forward(
        self,
        z_anc: torch.Tensor,       # [B, d_z]
        ctx_vec: torch.Tensor,     # [B, d_model]
        pred_base: torch.Tensor,   # [B, out_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (pred_final, alpha, pred_local)
          pred_final: [B, out_dim]
          alpha:      [B, 1]
          pred_local: [B, out_dim]
        """
        pred_local = self.W_local(ctx_vec)                       # [B, out_dim]
        alpha = torch.sigmoid(self.W_alpha(
            torch.cat([z_anc, ctx_vec], dim=-1)
        ))                                                        # [B, 1]
        pred_final = (1.0 - alpha) * pred_base + alpha * pred_local
        return pred_final, alpha, pred_local


def alpha_gate_loss(alpha: torch.Tensor, agree_score: torch.Tensor, lam: float = 0.05) -> torch.Tensor:
    """
    L_alpha = lam * MSE(alpha_squeezed, agree_score)

    alpha:        [B, 1] or [B]
    agree_score:  [B]   ∈ [0,1]
    """
    return lam * F.mse_loss(alpha.squeeze(-1), agree_score)
