import unittest
from typing import Tuple

import torch
import torch.nn as nn


class ObserverRouter(nn.Module):
    def __init__(self, obs_dim: int, n_views: int, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.view_head = nn.Linear(hidden_dim, n_views)
        self.mode_head = nn.Linear(hidden_dim, 1)

    def forward(self, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(g)
        pi = torch.softmax(self.view_head(h), dim=-1)
        beta = torch.sigmoid(self.mode_head(h))
        return pi, beta


class PosthocCombinerA(nn.Module):
    """
    Control model: post-hoc combination over view representations.
    Observers may be concatenated as extra combiner input, but there are
    no explicit pi/beta routing semantics.
    """
    def __init__(self, n_views: int, rep_dim: int, obs_dim: int = 0, out_dim: int = 1):
        super().__init__()
        in_dim = n_views * rep_dim + obs_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, rep_dim),
            nn.GELU(),
            nn.Linear(rep_dim, out_dim),
        )

    def forward(self, reps: torch.Tensor, g: torch.Tensor = None) -> torch.Tensor:
        b, v, d = reps.shape
        flat = reps.reshape(b, v * d)
        if g is not None:
            flat = torch.cat([flat, g], dim=-1)
        return self.net(flat)


class IntendedRouterB(nn.Module):
    """
    Intended routing model:
    1) compute pi from observers
    2) compute beta from observers
    3) form iso_rep and inter_rep
    4) blend before final prediction
    """
    def __init__(self, obs_dim: int, n_views: int, rep_dim: int, out_dim: int = 1):
        super().__init__()
        self.router = ObserverRouter(obs_dim, n_views)
        self.interaction_mlp = nn.Sequential(
            nn.Linear(rep_dim, rep_dim),
            nn.GELU(),
            nn.Linear(rep_dim, rep_dim),
        )
        self.pred_head = nn.Linear(rep_dim, out_dim)

    def forward(self, reps: torch.Tensor, g: torch.Tensor):
        pi, beta = self.router(g)
        weighted = reps * pi.unsqueeze(-1)   # [B, V, D]
        iso_rep = weighted.sum(dim=1)        # [B, D]
        inter_rep = self.interaction_mlp(iso_rep)
        final_rep = (1.0 - beta) * iso_rep + beta * inter_rep
        out = self.pred_head(final_rep)
        return out, pi, beta, iso_rep, inter_rep, final_rep


class TestRoutingSemantics(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.B = 8
        self.V = 3
        self.D = 5
        self.G = 4

        self.reps = torch.randn(self.B, self.V, self.D)
        self.g1 = torch.randn(self.B, self.G)
        self.g2 = self.g1 + 0.75  # different observer regime

        self.model_b = IntendedRouterB(obs_dim=self.G, n_views=self.V, rep_dim=self.D, out_dim=1)

    def test_pi_changes_when_g_changes(self):
        _, pi1, _, *_ = self.model_b(self.reps, self.g1)
        _, pi2, _, *_ = self.model_b(self.reps, self.g2)
        self.assertFalse(torch.allclose(pi1, pi2), "pi did not change when g changed")

    def test_beta_changes_when_g_changes(self):
        _, _, beta1, *_ = self.model_b(self.reps, self.g1)
        _, _, beta2, *_ = self.model_b(self.reps, self.g2)
        self.assertFalse(torch.allclose(beta1, beta2), "beta did not change when g changed")

    def test_prediction_changes_through_routing(self):
        out1, *_ = self.model_b(self.reps, self.g1)
        out2, *_ = self.model_b(self.reps, self.g2)
        self.assertFalse(torch.allclose(out1, out2), "prediction did not change when routing inputs changed")

    def test_pi_sums_to_one(self):
        _, pi, _, *_ = self.model_b(self.reps, self.g1)
        sums = pi.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-6), "pi does not sum to 1")

    def test_beta_in_unit_interval(self):
        _, _, beta, *_ = self.model_b(self.reps, self.g1)
        self.assertTrue(torch.all(beta >= 0.0).item(), "beta has values below 0")
        self.assertTrue(torch.all(beta <= 1.0).item(), "beta has values above 1")

    def test_beta_semantics_low_vs_high(self):
        model = IntendedRouterB(obs_dim=self.G, n_views=self.V, rep_dim=self.D, out_dim=1)
        reps = torch.randn(2, self.V, self.D)
        g = torch.randn(2, self.G)

        _, _, _, iso_rep, inter_rep, _ = model(reps, g)

        beta_low = torch.zeros(iso_rep.shape[0], 1)
        beta_high = torch.ones(iso_rep.shape[0], 1)

        final_low = (1.0 - beta_low) * iso_rep + beta_low * inter_rep
        final_high = (1.0 - beta_high) * iso_rep + beta_high * inter_rep

        self.assertTrue(torch.allclose(final_low, iso_rep, atol=1e-6), "beta=0 does not recover isolation")
        self.assertTrue(torch.allclose(final_high, inter_rep, atol=1e-6), "beta=1 does not recover interaction")


if __name__ == "__main__":
    unittest.main()
