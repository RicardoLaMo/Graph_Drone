"""
model.py — MV-TabR-GoRA: Multi-View label-in-KV retrieval model.

Architecture overview
---------------------
The model follows a 7-stage forward pass controlled by ablation flags.
Each stage adds exactly one new mechanism, making every A0→A6 gap
isolate a single causal contribution:

  A0  global kNN only (single FULL view), uniform attention + label in KV
  A1  per-view kNN, uniform routing (pi_v = 1/V for all v)
  A2  per-view kNN, sigma2_v soft routing with J-temperature scaling
  A3  A2  + T(z_i^v − z_j^v)  direction encoding in value
  A4  A3  + Q(q_i^v, q_j^v)   quality-pair encoding in value
  A5  A4  + learned routing MLP  (sigma2_v as input feature)
  A6  A5  + CrossViewMixer + β-gate  (isolation/interaction blend)

Validation evidence (from experiments/mv_gora_validation/reports/):
  • sigma2_v soft routing beats global kNN by +0.040 RMSE (H1 ✅)
  • Distance-weighting (→ T direction) beats plain means by +0.029 RMSE
  • J should scale routing temperature, NOT act as hard gate (Finding 4)
  • L2_stack_hgbr already ties TabM (0.4292); neural model has clear headroom
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    # Per-view input dimensions (CA: FULL=8, GEO=2, SOCIO=5, LOWRANK=4)
    view_dims: Dict[str, int] = field(default_factory=lambda: {
        "FULL": 8, "GEO": 2, "SOCIO": 5, "LOWRANK": 4
    })
    view_names: List[str] = field(default_factory=lambda: ["FULL", "GEO", "SOCIO", "LOWRANK"])
    # Core dimensions
    d_model: int = 64          # shared embedding dimension
    K: int = 24                # neighbours per view
    # Attention
    n_attn_heads: int = 4      # multi-head within-view attention
    dropout: float = 0.1
    # Ablation flags (set progressively A0 → A6)
    use_per_view: bool = True          # A1+: per-view encoders & routing (False = global FULL only)
    use_sigma2_routing: bool = False   # A2+: sigma2_v soft routing with J-temp
    use_direction_enc: bool = False    # A3+: T(z_i^v - z_j^v) in value
    use_quality_pair: bool = False     # A4+: Q(q_i^v, q_j^v) in value
    use_learned_routing: bool = False  # A5+: MLP routing (sigma2_v as input)
    use_cross_view_mixer: bool = False # A6:  CrossViewMixer + β-gate


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ViewEncoder(nn.Module):
    """Per-view linear encoder: x^(v) [*, d_in] → z^(v) [*, d_model]."""
    def __init__(self, d_in: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class LabelEmbed(nn.Module):
    """Scalar label → d_model embedding. Used to put y_j into the value."""
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y : [*, K]  →  [*, K, d_model]
        return self.norm(self.proj(y.unsqueeze(-1)))


class DirectionEncode(nn.Module):
    """
    T(z_i^v − z_j^v): anchor-relative direction encoding.
    Captures WHERE the anchor sits relative to each neighbor in view space.
    Validated: distance weighting (direction signal) adds +0.029 RMSE.
    """
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z_anc: torch.Tensor, z_nei: torch.Tensor) -> torch.Tensor:
        # z_anc: [B, d_model]  →  broadcast to [B, K, d_model]
        # z_nei: [B, K, d_model]
        diff = z_anc.unsqueeze(1) - z_nei        # [B, K, d_model]
        return self.norm(self.proj(diff))         # [B, K, d_model]


class QualityPairEncode(nn.Module):
    """
    Q(q_i^v, q_j^v): second-order trust signal.
    Both anchor and neighbor quality in the same view jointly determine
    how much to trust this particular (i,j,v) interaction.
    q_i, q_j are sigma2_v scalars (log-normalised, may be negative).
    """
    def __init__(self, d_model: int) -> None:
        super().__init__()
        # Input: [q_i, q_j, q_i*q_j, |q_i - q_j|] → 4 features
        self.proj = nn.Linear(4, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        q_anc: torch.Tensor,   # [B]    anchor sigma2_v for this view
        q_nei: torch.Tensor,   # [B, K] neighbor sigma2_v for this view
    ) -> torch.Tensor:
        qi = q_anc.unsqueeze(1).expand_as(q_nei)   # [B, K]
        qj = q_nei                                   # [B, K]
        feat = torch.stack([qi, qj, qi * qj, (qi - qj).abs()], dim=-1)  # [B, K, 4]
        return self.norm(self.proj(feat))            # [B, K, d_model]


class PerViewAttention(nn.Module):
    """
    Cross-attention reading over K neighbors in one view.

    Query  = z_i^v (anchor embedding in this view)
    Key    = z_j^v (neighbor embedding in this view)
    Value  = label_embed(y_j) [+ direction] [+ quality_pair]
             weighted by distance (log edge_weight as attention bias)
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        z_anc: torch.Tensor,        # [B, d_model] anchor embedding
        z_nei: torch.Tensor,        # [B, K, d_model] neighbor embeddings
        v_aug: torch.Tensor,        # [B, K, d_model] augmented value
        log_ew: torch.Tensor,       # [B, K] log edge weights (distance bias)
    ) -> torch.Tensor:
        B, K, _ = z_nei.shape
        H, D = self.n_heads, self.d_head

        # Query from anchor, Key from neighbors
        Q = self.W_Q(z_anc).view(B, 1, H, D).transpose(1, 2)         # [B, H, 1, D]
        K_ = self.W_K(z_nei).view(B, K, H, D).transpose(1, 2)        # [B, H, K, D]
        V_ = self.W_V(v_aug).view(B, K, H, D).transpose(1, 2)        # [B, H, K, D]

        # Scaled dot-product + distance bias
        scores = (Q @ K_.transpose(-2, -1)) / math.sqrt(D)           # [B, H, 1, K]
        # Broadcast log_ew across heads
        bias = log_ew.unsqueeze(1).unsqueeze(2)                       # [B, 1, 1, K]
        scores = scores + bias
        attn = self.drop(F.softmax(scores, dim=-1))                   # [B, H, 1, K]

        ctx = (attn @ V_).squeeze(2)                                  # [B, H, D]
        ctx = ctx.transpose(1, 2).contiguous().view(B, self.d_model)  # [B, d_model]
        ctx = self.W_out(ctx)
        return self.norm(ctx)                                          # [B, d_model]


class CrossViewMixer(nn.Module):
    """
    Self-attention over V view context vectors.
    Captures CROSS-VIEW interactions — which views agree, which diverge.
    Must be attention (not mean) to avoid collapsing to dumb average.
    """
    def __init__(self, n_views: int, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, ctx_stack: torch.Tensor) -> torch.Tensor:
        # ctx_stack: [B, V, d_model]
        out, _ = self.attn(ctx_stack, ctx_stack, ctx_stack)
        ctx_stack = self.norm(ctx_stack + out)
        ctx_stack = self.norm2(ctx_stack + self.ff(ctx_stack))
        return ctx_stack.mean(dim=1)   # [B, d_model]  pool across views


class ViewRouter(nn.Module):
    """
    Learned routing: [sigma2_v; z_global] → pi [B, V].

    Used for A5+ (use_learned_routing=True).
    For A2/A3/A4 (non-learned), call sigma2_routing() directly.

    J-temperature scaling (from validation: Finding 4):
        tau_v = 1 / (mean_J + eps)  — high agreement → sharp routing
    is applied OUTSIDE this module (in the main model forward) since it
    should always be applied regardless of learned vs non-learned routing.
    """
    def __init__(self, n_views: int, d_model: int) -> None:
        super().__init__()
        self.n_views = n_views
        hidden = max(n_views * 4, 16)
        # Input: [sigma2_v (V scalars)] + [z_global (d_model)]
        self.mlp = nn.Sequential(
            nn.Linear(n_views + d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_views),
        )

    def forward(
        self,
        sigma2_v: torch.Tensor,    # [B, V]
        z_global: torch.Tensor,    # [B, d_model]  mean of per-view anchor embeddings
    ) -> torch.Tensor:
        inp = torch.cat([sigma2_v, z_global], dim=-1)   # [B, V + d_model]
        return self.mlp(inp)                             # [B, V]  (raw logits → softmax outside)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MVTabrGoraModel(nn.Module):
    """
    Multi-View TabR-GoRA model.

    Ablation control (set via ModelConfig):
        A0: use_per_view=False, all others False
        A1: use_per_view=True,  all others False
        A2: use_per_view=True,  use_sigma2_routing=True
        A3: A2 + use_direction_enc=True
        A4: A3 + use_quality_pair=True
        A5: A4 + use_learned_routing=True
        A6: A5 + use_cross_view_mixer=True
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        V = len(cfg.view_names)

        # ---- Per-view encoders -------------------------------------------
        if cfg.use_per_view:
            self.view_encoders = nn.ModuleDict({
                name: ViewEncoder(cfg.view_dims[name], cfg.d_model)
                for name in cfg.view_names
            })
        else:
            # Global: single encoder on FULL view
            self.global_encoder = ViewEncoder(cfg.view_dims["FULL"], cfg.d_model)

        # ---- Value components --------------------------------------------
        self.label_embed = LabelEmbed(cfg.d_model)

        if cfg.use_direction_enc:
            self.direction_encs = nn.ModuleDict({
                name: DirectionEncode(cfg.d_model)
                for name in cfg.view_names
            })

        if cfg.use_quality_pair:
            self.quality_pair_encs = nn.ModuleDict({
                name: QualityPairEncode(cfg.d_model)
                for name in cfg.view_names
            })

        # ---- Value projection (combines label + optional components) ------
        # Count how many d_model slices will be concatenated into value
        n_value_slices = 1  # label_embed always present
        if cfg.use_direction_enc:
            n_value_slices += 1
        if cfg.use_quality_pair:
            n_value_slices += 1
        self.value_proj = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(n_value_slices * cfg.d_model, cfg.d_model),
                nn.LayerNorm(cfg.d_model),
            )
            for name in cfg.view_names
        })

        # ---- Per-view attention ------------------------------------------
        view_attn_names = cfg.view_names if cfg.use_per_view else ["FULL"]
        self.per_view_attn = nn.ModuleDict({
            name: PerViewAttention(cfg.d_model, cfg.n_attn_heads, cfg.dropout)
            for name in view_attn_names
        })

        # ---- Routing (A5+: learned; A2-A4: non-learned, no params here) --
        if cfg.use_learned_routing:
            self.router = ViewRouter(V, cfg.d_model)

        # ---- CrossViewMixer + β-gate (A6) --------------------------------
        if cfg.use_cross_view_mixer:
            self.cross_view_mixer = CrossViewMixer(V, cfg.d_model, cfg.n_attn_heads)
            # β-gate: [z_global; ctx_iso] → scalar in (0,1)
            self.beta_gate = nn.Sequential(
                nn.Linear(2 * cfg.d_model, cfg.d_model // 2),
                nn.GELU(),
                nn.Linear(cfg.d_model // 2, 1),
            )

        # ---- Task head ---------------------------------------------------
        self.task_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Zero-init the final task head bias; small std for label_embed."""
        nn.init.zeros_(self.task_head[-1].bias)
        nn.init.normal_(self.label_embed.proj.weight, std=0.01)

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _sigma2_routing(
        self,
        sigma2_v: torch.Tensor,   # [B, V]
        mean_J: torch.Tensor,     # [B]
    ) -> torch.Tensor:
        """
        Non-learned sigma2_v routing with J-temperature scaling (Finding 4):
            tau = 1 / (mean_J + eps)
            logit = -sigma2_v / tau
            pi = softmax(logit)
        High view agreement (high J) → tau small → sharp routing.
        Low view agreement (low J) → tau large → softer blend.
        """
        tau = 1.0 / (mean_J.unsqueeze(1) + EPS)      # [B, 1]
        logits = -sigma2_v / tau                       # [B, V]
        return F.softmax(logits, dim=-1)               # [B, V]

    def _uniform_routing(self, B: int, V: int, device: torch.device) -> torch.Tensor:
        return torch.full((B, V), 1.0 / V, device=device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        """
        batch keys (all Tensors on model device):
            x_anc         [B, F]
            x_nei         {name: [B, K, d_v]}   raw view features of neighbours
            y_nei         {name: [B, K]}         normalised labels of neighbours
            ew            {name: [B, K]}         edge weights (distance-based)
            sigma2_v_anc  [B, V]                 per-view label var of anchor
            sigma2_v_nei  {name: [B, K]}         per-view label var of each neighbour
            mean_J        [B]                    mean Jaccard (view agreement)

        Returns (pred [B], aux dict)
        """
        cfg = self.cfg
        V = len(cfg.view_names)
        B = batch["x_anc"].shape[0]
        device = batch["x_anc"].device
        aux = {}

        # ----------------------------------------------------------------
        # Stage 1: Per-view (or global) encoding
        # ----------------------------------------------------------------
        if cfg.use_per_view:
            z_anc_v: Dict[str, torch.Tensor] = {}
            z_nei_v: Dict[str, torch.Tensor] = {}
            for name in cfg.view_names:
                enc = self.view_encoders[name]
                z_anc_v[name] = enc(batch["x_anc_v"][name])          # [B, d_model]
                z_nei_v[name] = enc(batch["x_nei"][name])             # [B, K, d_model]
            # Global summary of anchor for routing input (A5+)
            z_global = torch.stack(list(z_anc_v.values()), dim=1).mean(dim=1)  # [B, d_model]
        else:
            # A0: single FULL view encoder
            z_anc_full = self.global_encoder(batch["x_anc_v"]["FULL"])    # [B, d_model]
            z_nei_full = self.global_encoder(batch["x_nei"]["FULL"])      # [B, K, d_model]
            z_anc_v = {"FULL": z_anc_full}
            z_nei_v = {"FULL": z_nei_full}
            z_global = z_anc_full

        active_views = cfg.view_names if cfg.use_per_view else ["FULL"]

        # ----------------------------------------------------------------
        # Stage 2: Routing weights pi [B, V_active]
        # ----------------------------------------------------------------
        sigma2_v_anc = batch["sigma2_v_anc"]   # [B, V]
        mean_J = batch["mean_J"]               # [B]

        if not cfg.use_per_view:
            # A0: single view, routing is trivially 1.0
            pi = torch.ones(B, 1, device=device)
        elif cfg.use_learned_routing:
            # A5+: learned MLP routing + J-temperature sharpening
            raw_logits = self.router(sigma2_v_anc, z_global)    # [B, V]
            tau = 1.0 / (mean_J.unsqueeze(1) + EPS)
            sharpened = raw_logits / tau                         # J-temp applied to learned logits
            pi = F.softmax(sharpened, dim=-1)                    # [B, V]
        elif cfg.use_sigma2_routing:
            # A2-A4: non-learned sigma2_v routing
            pi = self._sigma2_routing(sigma2_v_anc, mean_J)     # [B, V]
        else:
            # A1: uniform routing
            pi = self._uniform_routing(B, len(active_views), device)  # [B, V]

        aux["pi"] = pi.detach()

        # ----------------------------------------------------------------
        # Stage 3: Per-view value construction + attention
        # ----------------------------------------------------------------
        ctx_views: List[torch.Tensor] = []

        for v_idx, name in enumerate(active_views):
            ew_v = batch["ew"][name]                    # [B, K]
            y_nei_v = batch["y_nei"][name]              # [B, K]
            z_a = z_anc_v[name]                         # [B, d_model]
            z_n = z_nei_v[name]                         # [B, K, d_model]

            # Log edge weight as attention bias (distance → proximity signal)
            log_ew = torch.log(ew_v + EPS)              # [B, K]

            # Assemble value slices
            v_slices = [self.label_embed(y_nei_v)]      # [B, K, d_model]

            if cfg.use_direction_enc:
                t_enc = self.direction_encs[name](z_a, z_n)   # [B, K, d_model]
                v_slices.append(t_enc)

            if cfg.use_quality_pair:
                q_anc_v = sigma2_v_anc[:, v_idx]         # [B]
                q_nei_v = batch["sigma2_v_nei"][name]     # [B, K]
                q_enc = self.quality_pair_encs[name](q_anc_v, q_nei_v)  # [B, K, d_model]
                v_slices.append(q_enc)

            # Project concatenated value slices
            v_cat = torch.cat(v_slices, dim=-1)                    # [B, K, n_slices*d_model]
            v_aug = self.value_proj[name](v_cat)                   # [B, K, d_model]

            # Per-view cross-attention: anchor queries over neighbor KVs
            ctx_v = self.per_view_attn[name](z_a, z_n, v_aug, log_ew)  # [B, d_model]
            ctx_views.append(ctx_v)

        # ----------------------------------------------------------------
        # Stage 4: View aggregation
        # ----------------------------------------------------------------
        ctx_stack = torch.stack(ctx_views, dim=1)              # [B, V_active, d_model]

        # Isolation path: soft sigma2_v-routed blend
        ctx_iso = (pi.unsqueeze(-1) * ctx_stack).sum(dim=1)   # [B, d_model]

        if cfg.use_cross_view_mixer:
            # A6: interaction path through CrossViewMixer
            ctx_int = self.cross_view_mixer(ctx_stack)         # [B, d_model]
            # β-gate: [z_global; ctx_iso] → scalar blend coefficient
            beta = torch.sigmoid(
                self.beta_gate(torch.cat([z_global, ctx_iso], dim=-1))
            )                                                   # [B, 1]
            ctx_final = (1.0 - beta) * ctx_iso + beta * ctx_int  # [B, d_model]
            aux["beta_mean"] = float(beta.mean().item())
            aux["ctx_int_norm"] = float(ctx_int.norm(dim=-1).mean().item())
        else:
            ctx_final = ctx_iso

        # ----------------------------------------------------------------
        # Stage 5: Task head
        # ----------------------------------------------------------------
        pred = self.task_head(ctx_final).squeeze(-1)           # [B]
        return pred, aux


def ablation_config(
    name: str,
    view_dims: Optional[Dict[str, int]] = None,
    view_names: Optional[List[str]] = None,
    K: int = 24,
    d_model: int = 64,
) -> ModelConfig:
    """
    Construct ModelConfig for each ablation level by name.

    Valid names: A0, A1, A2, A3, A4, A5, A6
    """
    if view_dims is None:
        view_dims = {"FULL": 8, "GEO": 2, "SOCIO": 5, "LOWRANK": 4}
    if view_names is None:
        view_names = ["FULL", "GEO", "SOCIO", "LOWRANK"]

    base = dict(view_dims=view_dims, view_names=view_names, K=K, d_model=d_model)

    ablations = {
        "A0": dict(use_per_view=False, use_sigma2_routing=False,
                   use_direction_enc=False, use_quality_pair=False,
                   use_learned_routing=False, use_cross_view_mixer=False),
        "A1": dict(use_per_view=True,  use_sigma2_routing=False,
                   use_direction_enc=False, use_quality_pair=False,
                   use_learned_routing=False, use_cross_view_mixer=False),
        "A2": dict(use_per_view=True,  use_sigma2_routing=True,
                   use_direction_enc=False, use_quality_pair=False,
                   use_learned_routing=False, use_cross_view_mixer=False),
        "A3": dict(use_per_view=True,  use_sigma2_routing=True,
                   use_direction_enc=True, use_quality_pair=False,
                   use_learned_routing=False, use_cross_view_mixer=False),
        "A4": dict(use_per_view=True,  use_sigma2_routing=True,
                   use_direction_enc=True, use_quality_pair=True,
                   use_learned_routing=False, use_cross_view_mixer=False),
        "A5": dict(use_per_view=True,  use_sigma2_routing=True,
                   use_direction_enc=True, use_quality_pair=True,
                   use_learned_routing=True, use_cross_view_mixer=False),
        "A6": dict(use_per_view=True,  use_sigma2_routing=True,
                   use_direction_enc=True, use_quality_pair=True,
                   use_learned_routing=True, use_cross_view_mixer=True),
        # Fixed variants: A4/A5/A6 without Q encoding in value
        # (Q(q_i,q_j) found to hurt in first run; sigma2_v for routing only)
        "A4f": dict(use_per_view=True, use_sigma2_routing=True,
                    use_direction_enc=True, use_quality_pair=False,
                    use_learned_routing=True, use_cross_view_mixer=False),
        "A5f": dict(use_per_view=True, use_sigma2_routing=True,
                    use_direction_enc=True, use_quality_pair=False,
                    use_learned_routing=False, use_cross_view_mixer=True),
        "A6f": dict(use_per_view=True, use_sigma2_routing=True,
                    use_direction_enc=True, use_quality_pair=False,
                    use_learned_routing=True, use_cross_view_mixer=True),
    }
    if name not in ablations:
        raise ValueError(f"Unknown ablation '{name}'. Choose from {list(ablations)}")

    return ModelConfig(**{**base, **ablations[name]})
