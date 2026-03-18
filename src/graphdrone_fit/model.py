from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from typing import Optional, Union

from .config import GraphDroneConfig, SetRouterConfig
from .defer_integrator import IntegrationOutputs, integrate_predictions
from .expert_factory import (
    ExpertBuildSpec,
    IdentitySelectorAdapter,
    PortfolioExpertFactory,
    fit_portfolio_from_specs,
)
from .observers import calculate_kappa, calculate_lid
from .set_router import build_set_router
from .support_encoder import MomentSupportEncoder
from .token_builder import UniversalTokenBuilder
from .view_descriptor import ViewDescriptor


@dataclass(frozen=True)
class GraphDronePredictResult:
    predictions: np.ndarray
    expert_ids: tuple[str, ...]
    diagnostics: dict[str, object]


def _coerce_matrix(X: np.ndarray) -> np.ndarray:
    matrix = np.asarray(X, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {matrix.shape}")
    return matrix


class GraphDrone:
    def __init__(self, config: Optional[GraphDroneConfig] = None) -> None:
        self.config = config.validate() if config else GraphDroneConfig().validate()
        self._portfolio = None
        self._expert_factory = None
        self._router: Optional[torch.nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._support_encoder = MomentSupportEncoder()
        self._token_builder = UniversalTokenBuilder()
        self._X_tr: Optional[np.ndarray] = None  # stored for predict-time task_token
        self._y_tr: Optional[np.ndarray] = None  # stored for predict-time task_token
        self._train_views: dict[str, np.ndarray] = {}  # expert_id → view of training data for GORA

    def fit(self, X: np.ndarray, y: np.ndarray, expert_specs: Optional[tuple[ExpertBuildSpec, ...]] = None) -> "GraphDrone":
        X = np.asarray(X, dtype=np.float32)
        
        # Detect problem type if not fixed in config
        unique_y = np.unique(y)
        is_float_target = np.issubdtype(y.dtype, np.floating) and not np.all(np.mod(y, 1) == 0)
        
        if self.config.problem_type == "classification" or (
            self.config.problem_type != "regression"
            and not is_float_target
            and len(unique_y) < 50
        ):
            problem_type = "classification"
            # Respect config.n_classes if provided and > 1
            if self.config.n_classes > 1:
                n_classes = self.config.n_classes
            else:
                n_classes = int(unique_y.max() + 1)
            y = np.asarray(y, dtype=np.int64)
            base_model_kind = "foundation_classifier"
        else:
            problem_type = "regression"
            n_classes = 1
            y = np.asarray(y, dtype=np.float32)
            base_model_kind = "foundation_regressor"

        print(f"  -> Fitting GraphDrone for {problem_type} (n_classes={n_classes})...")

        # Internal validation split for router optimization
        from sklearn.model_selection import train_test_split
        stratify = y if problem_type == "classification" else None
        try:
            X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.1, random_state=42, stratify=stratify)
        except ValueError:
            # Fallback for very small classes
            X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.1, random_state=42)
        
        # Experts train on the full dataset; only the router uses the held-out val split.
        # This recovers the v1-width data-utilization strategy (100% for experts).
        full_matrix = _coerce_matrix(X)
        self.n_features_in_ = full_matrix.shape[1]
        self._X_tr = _coerce_matrix(X_tr)  # saved for task_token / GORA at predict time
        self._y_tr = y_tr                   # saved for task_token at predict time
        
        if expert_specs is None:
            full_idx = tuple(range(full_matrix.shape[1]))
            params = {"n_estimators": 8, "device": self.device}
            if problem_type == "classification":
                params["n_classes"] = n_classes
            expert_specs = (
                ExpertBuildSpec(
                    descriptor=ViewDescriptor(
                        expert_id=self.config.full_expert_id, family="FULL",
                        view_name="Full dataset", is_anchor=True, input_dim=full_matrix.shape[1], input_indices=full_idx
                    ),
                    model_kind=base_model_kind, 
                    input_adapter=IdentitySelectorAdapter(indices=full_idx),
                    model_params=params
                ),
            )
        else:
            # If user provided specs, ensure n_classes is injected for classifiers
            if problem_type == "classification":
                updated_specs = []
                for spec in expert_specs:
                    if "classifier" in spec.model_kind:
                        new_params = dict(spec.model_params)
                        new_params.setdefault("n_classes", n_classes)
                        updated_specs.append(ExpertBuildSpec(
                            descriptor=spec.descriptor, model_kind=spec.model_kind,
                            input_adapter=spec.input_adapter, model_params=new_params
                        ))
                    else:
                        updated_specs.append(spec)
                expert_specs = tuple(updated_specs)

        # Experts train on full data; router validates on the held-out 10% split.
        self._portfolio = fit_portfolio_from_specs(
            X_train=full_matrix, y_train=y, specs=expert_specs, full_expert_id=self.config.full_expert_id,
            n_jobs=1
        )
        self._expert_factory = PortfolioExpertFactory(self._portfolio)

        # Store per-expert training views for GORA geometric observers at predict time
        self._train_views = {}
        for spec in expert_specs:
            adapted = spec.input_adapter.transform(full_matrix)
            self._train_views[spec.descriptor.expert_id] = adapted

        # Router Training — build tokens with GORA observers for richer routing signal
        va_batch = self._expert_factory.predict_all(X_va)
        va_gora = self._compute_gora_obs(X_va, va_batch.descriptors)
        va_enc = self._support_encoder.encode(
            n_rows=len(X_va), descriptors=va_batch.descriptors, full_matrix=full_matrix,
            y_train=y_tr if problem_type == "regression" else None,
        )
        va_tokens = self._token_builder.build(
            predictions=va_batch.predictions,
            descriptors=va_batch.descriptors,
            full_expert_id=va_batch.full_expert_id,
            quality_scores=va_batch.quality_scores,
            support_encoding=va_enc,
            geometric_obs=va_gora,
        )
        
        token_dim = va_tokens.tokens.shape[-1]
        task_dim = va_tokens.task_token.shape[-1] if va_tokens.task_token is not None else 4
        
        # Size-Aware Routing Strategy
        if len(X_tr) < 500:
            print(f"  -> Small dataset detected (N={len(X_tr)}). Using Static Bootstrap Router to prevent overfitting.")
            self._router = build_set_router(SetRouterConfig(kind="bootstrap_full_only"), token_dim=token_dim, task_dim=task_dim).to(self.device)
            return self

        self._router = build_set_router(self.config.router, token_dim=token_dim, task_dim=task_dim).to(self.device)
        
        trainable_params = list(self._router.parameters())
        if not trainable_params:
            # Static router (BootstrapFullRouter) — no training needed
            print("  -> Router has no trainable parameters, skipping optimization.")
            return self

        optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
        
        # Initialize exploration bias to encourage early deferral (Smart Warm-start)
        if hasattr(self._router, 'exploration_bias'):
            with torch.no_grad():
                self._router.exploration_bias.fill_(1.0) # Start with ~73% defer rate boost

        best_loss = float('inf')
        patience = 25
        wait = 0
        
        print(f"  -> Optimizing Router for {problem_type} on {self.device} (Patience={patience})...")
        y_va_t = torch.tensor(y_va).to(self.device)
        v_preds_t = torch.tensor(va_batch.predictions).float().to(self.device) # [N, E, C]
        v_tokens_t = va_tokens.tokens.to(self.device)
        task_token_t = va_tokens.task_token.to(self.device) if va_tokens.task_token is not None else None

        for epoch in range(500):
            self._router.train()
            optimizer.zero_grad()
            out = self._router(v_tokens_t, full_index=va_batch.full_index, task_token=task_token_t)
            
            # Decay exploration bias
            if hasattr(self._router, 'exploration_bias'):
                with torch.no_grad():
                    self._router.exploration_bias.mul_(0.98)
            
            # Integration in training loop
            full_pred = v_preds_t[:, va_batch.full_index : va_batch.full_index + 1, :]
            
            # Mask out anchor from specialist weights
            spec_mask = torch.ones(v_preds_t.shape[1], device=self.device)
            spec_mask[va_batch.full_index] = 0
            
            masked_weights = out.specialist_weights * spec_mask.unsqueeze(0)
            weight_sum = masked_weights.sum(dim=1, keepdim=True)
            norm_spec_weights = masked_weights / torch.where(weight_sum > 0, weight_sum, torch.ones_like(weight_sum))
            
            spec_pred = (norm_spec_weights.unsqueeze(-1) * v_preds_t).sum(dim=1, keepdim=True)
            
            # Residual Probability Blending (RPB):
            # specialists nudging the anchor: anchor + defer * (specialists - anchor)
            if problem_type == "regression":
                integ = full_pred + out.defer_prob.unsqueeze(-1) * (spec_pred - full_pred)
            else:
                integ = (1 - out.defer_prob.unsqueeze(-1)) * full_pred + out.defer_prob.unsqueeze(-1) * spec_pred
            
            integ = integ.squeeze(1)  # [N, C]
            full_pred_sq = full_pred.squeeze(1) # [N, C]

            if problem_type == "classification":
                main_loss = torch.nn.functional.nll_loss(torch.log(torch.clamp(integ, min=1e-8)), y_va_t)
                anchor_loss = torch.nn.functional.nll_loss(torch.log(torch.clamp(full_pred_sq, min=1e-8)), y_va_t)
            else:
                main_loss = torch.nn.functional.mse_loss(integ.squeeze(), y_va_t)
                anchor_loss = torch.nn.functional.mse_loss(full_pred_sq.squeeze(), y_va_t)
            
            # Anchor-Aware Residual Penalty
            residual_penalty = torch.relu(main_loss - anchor_loss)
            defer_penalty = out.defer_prob.mean() * 0.05
            
            loss = main_loss + 2.0 * residual_penalty + defer_penalty
                
            loss.backward(); optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
        return self

    def _compute_gora_obs(self, X: np.ndarray, descriptors: tuple) -> torch.Tensor:
        """Compute per-expert GORA geometric observers (kappa + LID) for each query point."""
        from sklearn.neighbors import NearestNeighbors
        all_obs = []
        for d in descriptors:
            X_tr_v = self._train_views.get(d.expert_id)
            if X_tr_v is None or len(X_tr_v) < 2:
                all_obs.append(np.zeros((len(X), 2), dtype=np.float32))
                continue
            X_v = X[:, list(d.input_indices)] if d.input_indices else X
            k = min(getattr(d, 'preferred_k', 15), len(X_tr_v) - 1)
            knn = NearestNeighbors(n_neighbors=k).fit(X_tr_v)
            dists, indices = knn.kneighbors(X_v)
            kappa = calculate_kappa(X_tr_v, indices).reshape(-1, 1)
            lid = calculate_lid(dists).reshape(-1, 1)
            all_obs.append(np.concatenate([kappa, lid], axis=1))
        return torch.tensor(np.stack(all_obs, axis=1), dtype=torch.float32)

    def predict(self, X: np.ndarray, return_diagnostics: bool = False) -> Union[np.ndarray, GraphDronePredictResult]:
        X = np.asarray(X, dtype=np.float32)
        matrix = _coerce_matrix(X)
        batch = self._expert_factory.predict_all(matrix)
        
        # Pass training matrix/target so task_token reflects dataset statistics from training
        support_enc = self._support_encoder.encode(
            n_rows=matrix.shape[0], descriptors=batch.descriptors, 
            full_matrix=self._X_tr if self._X_tr is not None else matrix,
            y_train=self._y_tr
        )
        gora_obs = self._compute_gora_obs(matrix, batch.descriptors)
        tokens = self._token_builder.build(
            predictions=batch.predictions,
            descriptors=batch.descriptors,
            full_expert_id=batch.full_expert_id,
            quality_scores=batch.quality_scores,
            support_encoding=support_enc,
            geometric_obs=gora_obs,
        )
        
        self._router.eval()
        with torch.no_grad():
            token_tensor = tokens.tokens.to(self.device)
            task_token_t = tokens.task_token.to(self.device) if tokens.task_token is not None else None
            router_out = self._router(token_tensor, full_index=batch.full_index, task_token=task_token_t)
            
            pred_tensor = torch.as_tensor(batch.predictions, dtype=torch.float32).to(self.device)
            if pred_tensor.ndim == 2: pred_tensor = pred_tensor.unsqueeze(-1)
            
            full_pred = pred_tensor[:, batch.full_index : batch.full_index + 1, :]
            
            # Mask out anchor from specialist weights
            spec_mask = torch.ones(pred_tensor.shape[1], device=self.device)
            spec_mask[batch.full_index] = 0
            
            masked_weights = router_out.specialist_weights * spec_mask.unsqueeze(0)
            weight_mass = masked_weights.sum(dim=1, keepdim=True)
            norm_spec_weights = masked_weights / torch.where(weight_mass > 0, weight_mass, torch.ones_like(weight_mass))
            
            spec_pred = (norm_spec_weights.unsqueeze(-1) * pred_tensor).sum(dim=1, keepdim=True)
            
            if self.config.problem_type == "regression":
                blended = full_pred + router_out.defer_prob.unsqueeze(-1) * (spec_pred - full_pred)
            else:
                blended = (1.0 - router_out.defer_prob.unsqueeze(-1)) * full_pred + router_out.defer_prob.unsqueeze(-1) * spec_pred
            
            result = blended.squeeze(1)
            if result.shape[1] == 1: result = result.squeeze(1)
            
            integration = IntegrationOutputs(
                predictions=result.detach().cpu().numpy().astype(np.float32),
                normalized_weights=norm_spec_weights.detach().cpu().numpy().astype(np.float32),
                defer_prob=router_out.defer_prob.detach().cpu().numpy().astype(np.float32),
                diagnostics={
                    "mean_defer_prob": float(router_out.defer_prob.mean().item()),
                    "router_kind": router_out.router_kind
                },
            )
            
        if return_diagnostics:
            return GraphDronePredictResult(
                predictions=integration.predictions,
                expert_ids=batch.expert_ids,
                diagnostics=integration.diagnostics
            )
        return integration.predictions
