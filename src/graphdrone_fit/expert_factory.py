from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .portfolio_loader import LoadedPortfolio
from .view_descriptor import ViewDescriptor


@dataclass(frozen=True)
class ExpertPredictionBatch:
    expert_ids: tuple[str, ...]
    descriptors: tuple[ViewDescriptor, ...]
    predictions: np.ndarray
    full_expert_id: str
    full_index: int


class PortfolioExpertFactory:
    def __init__(self, portfolio: LoadedPortfolio) -> None:
        self.portfolio = portfolio.validate()
        self.expert_ids = self.portfolio.expert_order
        self.descriptors = self.portfolio.descriptors
        self.full_expert_id = self.portfolio.full_expert_id
        self.full_index = self.expert_ids.index(self.full_expert_id)

    def predict_all(self, X: np.ndarray) -> ExpertPredictionBatch:
        preds = [
            self.portfolio.experts[expert_id].predict(X)
            for expert_id in self.expert_ids
        ]
        stacked = np.column_stack(preds).astype(np.float32)
        return ExpertPredictionBatch(
            expert_ids=self.expert_ids,
            descriptors=self.descriptors,
            predictions=stacked,
            full_expert_id=self.full_expert_id,
            full_index=self.full_index,
        )
