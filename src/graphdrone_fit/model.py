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
    Optimized for data-retention, H200 parallelization, and GORA Geometric Awareness.
    """
    def __init__(self, config: GraphDroneConfig) -> None:
        self.config = config.validate()
        self._expert_factory: Optional[PortfolioExpertFactory] = None
        self._token_builder = UniversalTokenBuilder()
        self._support_encoder = MomentSupportEncoder()
        self._router: Optional[torch.nn.Module] = None
        self._train_views: dict[str, np.ndarray] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X: np.ndarray, y: np.ndarray, expert_specs: Optional[tuple[ExpertBuildSpec, ...]] = None, problem_type: Optional[str] = None) -> "GraphDrone":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        # 1. Expert Fitting (100% DATA UTILIZATION)
        matrix = _coerce_matrix(X)
        self.n_features_in_ = matrix.shape[1]
        
        if expert_specs is None:
            full_idx = tuple(range(self.n_features_in_))
            params = {"n_estimators": 8, "device": self.device}
            expert_specs = (
                ExpertBuildSpec(
                    descriptor=ViewDescriptor(
                        expert_id=self.config.full_expert_id, family="FULL", 
                        view_name="Full dataset", is_anchor=True, input_dim=self.n_features_in_, input_indices=full_idx
                    ),
                    model_kind="foundation_regressor", 
                    input_adapter=IdentitySelectorAdapter(indices=full_idx),
                    model_params=params
                ),
            )

        # Auto-detect problem type if not provided
        unique_y = np.unique(y)
        if problem_type is None:
            problem_type = "binary" if (len(unique_y) == 2 and set(unique_y.tolist()) <= {0.0, 1.0}) else "regression"
        self._problem_type = problem_type

        print(f"  -> Fitting specialists on {len(X)} samples...")
        self._portfolio = fit_portfolio_from_specs(
            X_train=matrix, y_train=y, specs=expert_specs, full_expert_id=self.config.full_expert_id
        )
        self._expert_factory = PortfolioExpertFactory(self._portfolio)
        
        # Store training views for GORA observers
        for spec in expert_specs:
            fitted_adapter = spec.input_adapter.fit(matrix)
            self._train_views[spec.descriptor.expert_id] = fitted_adapter.transform(matrix)

        # 2. Router Optimization (Internal 10% Split)
        from sklearn.model_selection import train_test_split
        _, X_va, _, y_va = train_test_split(X, y, test_size=0.2, random_state=42)
        
        va_batch = self._expert_factory.predict_all(X_va)
        va_enc = self._support_encoder.encode(n_rows=len(X_va), descriptors=va_batch.descriptors)
        
        # Compute GORA observers for VAL
        va_gora = self._compute_gora_obs(X_va, va_batch.descriptors)
        
        va_tokens = self._token_builder.build(
            predictions=va_batch.predictions, descriptors=va_batch.descriptors,
            full_expert_id=va_batch.full_expert_id, support_encoding=va_enc,
            geometric_obs=va_gora
        )
        
        token_dim = va_tokens.tokens.shape[-1]
        self._router = build_set_router(self.config.router, token_dim=token_dim).to(self.device)
        optimizer = torch.optim.Adam(self._router.parameters(), lr=1e-3)
        
        best_loss = float('inf')
        patience = 25
        wait = 0
        
        is_binary = (self._problem_type == "binary")
        print(f"  -> Optimizing Router on {self.device} (Patience={patience}, loss={'BCE' if is_binary else 'MSE'})...")
        y_va_t = torch.tensor(y_va).float().to(self.device)
        v_preds_t = torch.tensor(va_batch.predictions).float().to(self.device)
        v_tokens_t = va_tokens.tokens.to(self.device)

        for epoch in range(500):
            self._router.train()
            optimizer.zero_grad()
            out = self._router(v_tokens_t, full_index=va_batch.full_index)
            integ = (1 - out.defer_prob) * v_preds_t[:, va_batch.full_index:va_batch.full_index+1] + \
                    out.defer_prob * (out.specialist_weights * v_preds_t).sum(dim=1, keepdim=True)
            if is_binary:
                loss = torch.nn.functional.binary_cross_entropy(
                    integ.squeeze().nan_to_num(nan=0.5, posinf=1.0, neginf=0.0).clamp(1e-6, 1 - 1e-6), y_va_t
                )
            else:
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

    def _compute_gora_obs(self, X: np.ndarray, descriptors: tuple[ViewDescriptor, ...]) -> torch.Tensor:
        from sklearn.neighbors import NearestNeighbors
        from .observers import calculate_kappa, calculate_lid
        
        all_obs = []
        for d in descriptors:
            X_tr_v = self._train_views[d.expert_id]
            # subselect logic
            X_v = X[:, list(d.input_indices)] if d.input_indices else X
            
            knn = NearestNeighbors(n_neighbors=d.preferred_k).fit(X_tr_v)
            dists, indices = knn.kneighbors(X_v)
            # Use X_tr_v here, NOT X_v, because indices refer to X_tr_v
            kappa = calculate_kappa(X_tr_v, indices).reshape(-1, 1)
            lid = calculate_lid(dists).reshape(-1, 1)
            all_obs.append(np.concatenate([kappa, lid], axis=1))
            
        return torch.tensor(np.stack(all_obs, axis=1), dtype=torch.float32)

    def predict(self, X: np.ndarray, return_diagnostics: bool = False) -> Union[np.ndarray, GraphDronePredictResult]:
        X = np.asarray(X, dtype=np.float32)
        matrix = _coerce_matrix(X)
        batch = self._expert_factory.predict_all(matrix)
        
        # Internalized Support/Token/GORA workflow
        support_enc = self._support_encoder.encode(n_rows=matrix.shape[0], descriptors=batch.descriptors)
        gora_obs = self._compute_gora_obs(matrix, batch.descriptors)
        
        tokens = self._token_builder.build(
            predictions=batch.predictions,
            descriptors=batch.descriptors,
            full_expert_id=batch.full_expert_id,
            support_encoding=support_enc,
            geometric_obs=gora_obs
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
