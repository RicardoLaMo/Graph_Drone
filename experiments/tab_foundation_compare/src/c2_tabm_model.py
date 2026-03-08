from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def load_tabm_module(tabm_root: Path) -> Any:
    tabm_path = tabm_root / "tabm.py"
    spec = importlib.util.spec_from_file_location("tabm_local", tabm_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load tabm module from {tabm_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class DecoderConfig:
    n_num_features: int
    k: int = 32
    d_block: int = 400
    n_blocks: int = 3
    dropout: float = 0.20769705860329654


class MeanTabMRegressor(nn.Module):
    def __init__(self, tabm_module: Any, config: DecoderConfig) -> None:
        super().__init__()
        self.model = tabm_module.TabM.make(
            n_num_features=config.n_num_features,
            d_out=1,
            k=config.k,
            n_blocks=config.n_blocks,
            d_block=config.d_block,
            dropout=config.dropout,
            arch_type="tabm",
        )

    def forward(self, x_num: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        head_preds = self.model(x_num).squeeze(-1)
        pred = head_preds.mean(dim=1)
        return pred, {"head_preds": head_preds}


class GatedTabMRegressor(nn.Module):
    def __init__(self, tabm_module: Any, config: DecoderConfig) -> None:
        super().__init__()
        self.backbone_model = tabm_module.TabM.make(
            n_num_features=config.n_num_features,
            d_out=None,
            k=config.k,
            n_blocks=config.n_blocks,
            d_block=config.d_block,
            dropout=config.dropout,
            arch_type="tabm",
        )
        self.head_output = tabm_module.LinearEnsemble(config.d_block, 1, k=config.k)
        self.gate = nn.Sequential(
            nn.Linear(config.k, max(config.k // 2, 8)),
            nn.GELU(),
            nn.Linear(max(config.k // 2, 8), config.k),
        )
        self.residual = nn.Sequential(
            nn.Linear(config.d_block, config.d_block // 2),
            nn.GELU(),
            nn.Linear(config.d_block // 2, 1),
        )

    def forward(self, x_num: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.backbone_model(x_num)
        head_preds = self.head_output(hidden).squeeze(-1)
        gate_logits = self.gate(head_preds)
        gate = torch.softmax(gate_logits, dim=1)
        weighted = (gate * head_preds).sum(dim=1)
        residual = self.residual(hidden.mean(dim=1)).squeeze(-1)
        pred = weighted + residual
        return pred, {"head_preds": head_preds, "gate": gate, "hidden": hidden}
