import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
V4_SRC = REPO_ROOT / "experiments" / "mq_gora_v4" / "shared" / "src"
if str(V4_SRC) not in sys.path:
    sys.path.insert(0, str(V4_SRC))

from row_transformer_v4 import SplitTrackRouter, blend_iso_interaction


def test_split_track_router_returns_pi_beta_tau():
    torch.manual_seed(7)
    router = SplitTrackRouter(
        obs_dim=4,
        n_heads=3,
        n_views=2,
        d_z=5,
        d_model=6,
        has_z=True,
        has_label=True,
        has_ctx=True,
        hidden=8,
    )
    g = torch.randn(5, 4)
    z = torch.randn(5, 5)
    label_ctx = torch.randn(5, 5)
    ctx = torch.randn(5, 6)

    pi, beta, tau = router(g=g, z_anc=z, label_ctx_vec=label_ctx, ctx_vec=ctx)

    assert pi.shape == (5, 3, 2)
    assert beta.shape == (5, 3)
    assert tau.shape == (3,)
    assert torch.allclose(pi.sum(dim=-1), torch.ones(5, 3), atol=1e-6)
    assert torch.all((beta >= 0.0) & (beta <= 1.0))


def test_split_track_router_changes_beta_when_observers_change():
    torch.manual_seed(11)
    router = SplitTrackRouter(obs_dim=4, n_heads=2, n_views=3, hidden=8)
    g1 = torch.randn(6, 4)
    g2 = g1 + 0.75

    _, beta1, _ = router(g=g1)
    _, beta2, _ = router(g=g2)

    assert not torch.allclose(beta1, beta2)


def test_blend_iso_interaction_respects_beta_extremes():
    iso_heads = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]]],
        dtype=torch.float32,
    )
    inter_rep = torch.tensor([[10.0, 20.0]], dtype=torch.float32)

    low = blend_iso_interaction(iso_heads, inter_rep, torch.zeros(1, 2))
    high = blend_iso_interaction(iso_heads, inter_rep, torch.ones(1, 2))

    assert torch.allclose(low, iso_heads.mean(dim=1), atol=1e-6)
    assert torch.allclose(high, inter_rep, atol=1e-6)
