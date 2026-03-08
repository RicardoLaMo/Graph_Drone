from pathlib import Path

import torch

from experiments.tab_foundation_compare.src.c2_tabm_model import (
    DecoderConfig,
    GatedTabMRegressor,
    MeanTabMRegressor,
    load_tabm_module,
)


TABM_ROOT = Path("/private/tmp/tabm_clone_inspect_20260308")


def test_mean_tabm_output_shapes():
    tabm_module = load_tabm_module(TABM_ROOT)
    model = MeanTabMRegressor(tabm_module, DecoderConfig(n_num_features=8, k=4, d_block=32, n_blocks=2))
    x = torch.randn(16, 8)
    pred, aux = model(x)
    assert pred.shape == (16,)
    assert aux["head_preds"].shape == (16, 4)


def test_gated_tabm_gate_is_row_normalized():
    tabm_module = load_tabm_module(TABM_ROOT)
    model = GatedTabMRegressor(tabm_module, DecoderConfig(n_num_features=8, k=4, d_block=32, n_blocks=2))
    x = torch.randn(10, 8)
    pred, aux = model(x)
    assert pred.shape == (10,)
    assert aux["head_preds"].shape == (10, 4)
    assert aux["gate"].shape == (10, 4)
    assert torch.allclose(aux["gate"].sum(dim=1), torch.ones(10), atol=1e-5)
