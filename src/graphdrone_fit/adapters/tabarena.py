from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from tabarena.benchmark.models.wrapper.abstract_class import AbstractExecModel
from graphdrone_fit import GraphDrone, GraphDroneConfig, SetRouterConfig, ExpertBuildSpec, ViewDescriptor, IdentitySelectorAdapter

# Monkey-patch GpuMemoryTracker to disable GPU memory tracking
# This avoids CUDA device-side assert errors caused by hardcoded device=0
import tabarena.utils.memory_utils as mem_utils
_original_gpu_tracker_init = mem_utils.GpuMemoryTracker.__init__

def _patched_gpu_tracker_init(self, device=0, interval=0.05):
    _original_gpu_tracker_init(self, device=device, interval=interval)
    # Force disable GPU memory tracking to avoid CUDA sync errors
    self.enabled = False

mem_utils.GpuMemoryTracker.__init__ = _patched_gpu_tracker_init

class GraphDroneTabArenaAdapter(AbstractExecModel):
    """
    Enhanced Adapter for TabArena with Hybrid Prior support.
    """
    def __init__(self, *args, n_estimators: int = 8, router_kind: str = "noise_gate_router",
                 use_gora: bool = True, use_hybrid_prior: bool = False, **kwargs):
        # Extract device from kwargs if present (TabArena may pass it)
        kwargs.pop('device', None)
        super().__init__(*args, **kwargs)
        self.n_estimators = n_estimators
        self.router_kind = router_kind
        self.use_gora = use_gora
        self.use_hybrid_prior = use_hybrid_prior
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None):
        n_features = X.shape[1]
        mid = n_features // 2
        
        if self.problem_type in ["binary", "multiclass"]:
            base_kind = "foundation_classifier"
            hybrid_kind = "xgboost_classifier"
        else:
            base_kind = "foundation_regressor"
            hybrid_kind = "xgboost_regressor"

        full_idx = tuple(range(n_features))
        v1_idx = tuple(range(mid))
        v2_idx = tuple(range(mid, n_features))
        
        params = {"n_estimators": self.n_estimators, "device": self.device}
        
        # HYBRID LOGIC: 
        # Specialist 1: Foundation
        # Specialist 2: XGBoost (The "Cleaner" structural prior)
        specs = (
            ExpertBuildSpec(
                descriptor=ViewDescriptor(expert_id="FULL", family="FULL", view_name="Full", is_anchor=True, input_dim=n_features, input_indices=full_idx),
                model_kind=base_kind, input_adapter=IdentitySelectorAdapter(indices=full_idx), model_params=params
            ),
            ExpertBuildSpec(
                descriptor=ViewDescriptor(expert_id="V1", family="structural_subspace", view_name="V1", input_dim=len(v1_idx), input_indices=v1_idx),
                model_kind=base_kind, input_adapter=IdentitySelectorAdapter(indices=v1_idx), model_params=params
            ),
            ExpertBuildSpec(
                descriptor=ViewDescriptor(expert_id="V2", family="structural_subspace", view_name="V2", input_dim=len(v2_idx), input_indices=v2_idx),
                model_kind=hybrid_kind if self.use_hybrid_prior else base_kind, 
                input_adapter=IdentitySelectorAdapter(indices=v2_idx), model_params=params
            )
        )

        config = GraphDroneConfig(full_expert_id="FULL", router=SetRouterConfig(kind=self.router_kind))
        self.model = GraphDrone(config)
        
        if not self.use_gora:
            self.model._compute_gora_obs = lambda X, descriptors: torch.zeros((X.shape[0], len(descriptors), 2), device=self.model.device)

        self.model.fit(X.values.astype(np.float32), y.values.astype(np.float32), expert_specs=specs)
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        preds = self.model.predict(X.values.astype(np.float32))
        return pd.Series(preds, index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        probas = self.model.predict(X.values.astype(np.float32))
        if self.problem_type == "binary":
            if probas.ndim == 1:
                df_probas = pd.DataFrame({0: 1 - probas, 1: probas}, index=X.index)
            else:
                df_probas = pd.DataFrame(probas, index=X.index)
        else:
            df_probas = pd.DataFrame(probas, index=X.index)
        return df_probas
