from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from typing import Optional, Union

from .config import GraphDroneConfig
from .defer_integrator import IntegrationOutputs, integrate_predictions
from .expert_factory import ExpertBuildSpec, ExpertPredictionBatch, PortfolioExpertFactory, fit_portfolio_from_specs, IdentitySelectorAdapter
from .portfolio_loader import LoadedPortfolio, load_portfolio
from .set_router import build_set_router
from .support_encoder import MomentSupportEncoder, SupportEncoding
from .token_builder import UniversalTokenBuilder, TokenBatch
from .view_descriptor import ViewDescriptor

@dataclass(frozen=True)
class GraphDronePredictResult:
    predictions: np.ndarray
    diagnostics: dict[str, object]
    expert_ids: tuple[str, ...]

def _coerce_matrix(X: np.ndarray) -> np.ndarray:
    matrix = np.asarray(X, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {matrix.shape}")
    return matrix

class GraphDrone:
    """
    High-Performance GraphDrone Meta-Model.
    Optimized for data-retention and H200 parallelization.
    """
    def __init__(self, config: GraphDroneConfig) -> None:
        self.config = config.validate()
        self._expert_factory: Optional[PortfolioExpertFactory] = None
        self._token_builder = UniversalTokenBuilder()
        self._support_encoder = MomentSupportEncoder()
        self._router: Optional[torch.nn.Module] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X: np.ndarray, y: np.ndarray, expert_specs: Optional[tuple[ExpertBuildSpec, ...]] = None) -> "GraphDrone":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        matrix = _coerce_matrix(X)
        
        # 1. Expert Fitting (100% DATA UTILIZATION)
        # We fit experts on the full set to ensure base power matches/exceeds baselines.
        if expert_specs is None:
            full_idx = tuple(range(matrix.shape[1]))
            params = {"n_estimators": 8, "device": self.device} # Higher fidelity default
            expert_specs = (
                ExpertBuildSpec(
                    descriptor=ViewDescriptor(
                        expert_id=self.config.full_expert_id, family="FULL", 
                        view_name="Full dataset", is_anchor=True, input_dim=matrix.shape[1], input_indices=full_idx
                    ),
                    model_kind="foundation_regressor", 
                    input_adapter=IdentitySelectorAdapter(indices=full_idx),
                    model_params=params
                ),
            )

        print(f"  -> Fitting specialists on {len(X)} samples...")
        self._portfolio = fit_portfolio_from_specs(
            X_train=matrix, y_train=y, specs=expert_specs, full_expert_id=self.config.full_expert_id
        )
        self._expert_factory = PortfolioExpertFactory(self._portfolio)
        
        # 2. Router Optimization (Internal 90/10 Split)
        # We use a smaller split here because the experts are already strong.
        from sklearn.model_selection import train_test_split
        _, X_va, _, y_va = train_test_split(X, y, test_size=0.1, random_state=42)
        
        va_batch = self._expert_factory.predict_all(X_va)
        va_enc = self._support_encoder.encode(n_rows=len(X_va), descriptors=va_batch.descriptors)
        va_tokens = self._token_builder.build(
            predictions=va_batch.predictions, descriptors=va_batch.descriptors,
            full_expert_id=va_batch.full_expert_id, support_encoding=va_enc
        )
        
        token_dim = va_tokens.tokens.shape[-1]
        self._router = build_set_router(self.config.router, token_dim=token_dim).to(self.device)
        optimizer = torch.optim.Adam(self._router.parameters(), lr=1e-3)
        
        best_loss = float('inf')
        patience = 25 # Increased patience for better convergence
        wait = 0
        
        print(f"  -> Optimizing Router on {self.device} (Patience={patience})...")
        y_va_t = torch.tensor(y_va).float().to(self.device)
        v_preds_t = torch.tensor(va_batch.predictions).float().to(self.device)
        v_tokens_t = va_tokens.tokens.to(self.device)

        for epoch in range(500):
            self._router.train()
            optimizer.zero_grad()
            out = self._router(v_tokens_t, full_index=va_batch.full_index)
            integ = (1 - out.defer_prob) * v_preds_t[:, va_batch.full_index:va_batch.full_index+1] + \
                    out.defer_prob * (out.specialist_weights * v_preds_t).sum(dim=1, keepdim=True)
            loss = torch.nn.functional.mse_loss(integ.squeeze(), y_va_t)
            loss.backward(); optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
        return self

    def predict(self, X: np.ndarray, return_diagnostics: bool = False) -> Union[np.ndarray, GraphDronePredictResult]:
        X = np.asarray(X, dtype=np.float32)
        matrix = _coerce_matrix(X)
        batch = self._expert_factory.predict_all(matrix)
        
        support_enc = self._support_encoder.encode(n_rows=matrix.shape[0], descriptors=batch.descriptors)
        tokens = self._token_builder.build(
            predictions=batch.predictions,
            descriptors=batch.descriptors,
            full_expert_id=batch.full_expert_id,
            support_encoding=support_enc
        )
        
        self._router.eval()
        with torch.no_grad():
            token_tensor = tokens.tokens.to(self.device)
            router_out = self._router(token_tensor, full_index=batch.full_index)
            integration = integrate_predictions(expert_predictions=batch.predictions, router_outputs=router_out)
            
        if return_diagnostics:
            return GraphDronePredictResult(
                predictions=integration.predictions,
                expert_ids=batch.expert_ids,
                diagnostics=integration.diagnostics
            )
        return integration.predictions
