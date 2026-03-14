from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from tabarena.benchmark.models.wrapper.abstract_class import AbstractExecModel
from graphdrone_fit import GraphDrone, GraphDroneConfig, SetRouterConfig, ExpertBuildSpec, ViewDescriptor, IdentitySelectorAdapter

class GraphDroneTabArenaAdapter(AbstractExecModel):
    """
    Adapter to run GraphDrone within the TabArena benchmarking framework.
    """
    def __init__(self, *args, n_estimators: int = 8, router_kind: str = "noise_gate_router", **kwargs):
        super().__init__(*args, **kwargs)
        self.n_estimators = n_estimators
        self.router_kind = router_kind
        self.model = None
        self.imputer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _to_array(self, X: pd.DataFrame) -> np.ndarray:
        """Convert DataFrame to float32 array, imputing NaNs with training medians."""
        arr = X.values.astype(np.float32)
        if self.imputer is not None:
            arr = self.imputer.transform(arr)
        return arr

    def _fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None):
        n_features = X.shape[1]
        mid = n_features // 2
        
        # Determine model kind based on problem type
        if self.problem_type in ["binary", "multiclass"]:
            model_kind = "foundation_classifier"
        else:
            model_kind = "foundation_regressor"

        # Define default 3-view portfolio for TabArena
        full_idx = tuple(range(n_features))
        v1_idx = tuple(range(mid))
        v2_idx = tuple(range(mid, n_features))
        adaptive_k = int(np.clip(int(np.sqrt(len(X)) / 2), 5, 30))

        params = {"n_estimators": self.n_estimators, "device": self.device}

        specs = (
            ExpertBuildSpec(
                descriptor=ViewDescriptor(
                    expert_id="FULL", family="FULL", view_name="Full dataset",
                    is_anchor=True, input_dim=n_features, input_indices=full_idx,
                    preferred_k=adaptive_k
                ),
                model_kind=model_kind,
                input_adapter=IdentitySelectorAdapter(indices=full_idx),
                model_params=params
            ),
            ExpertBuildSpec(
                descriptor=ViewDescriptor(
                    expert_id="V1", family="structural_subspace", view_name="First half features",
                    input_dim=len(v1_idx), input_indices=v1_idx,
                    preferred_k=adaptive_k
                ),
                model_kind=model_kind,
                input_adapter=IdentitySelectorAdapter(indices=v1_idx),
                model_params=params
            ),
            ExpertBuildSpec(
                descriptor=ViewDescriptor(
                    expert_id="V2", family="structural_subspace", view_name="Second half features",
                    input_dim=len(v2_idx), input_indices=v2_idx,
                    preferred_k=adaptive_k
                ),
                model_kind=model_kind,
                input_adapter=IdentitySelectorAdapter(indices=v2_idx),
                model_params=params
            )
        )

        config = GraphDroneConfig(
            full_expert_id="FULL",
            router=SetRouterConfig(kind=self.router_kind)
        )
        
        # Fit imputer on training data to handle datasets with missing values
        X_arr = X.values.astype(np.float32)
        if np.isnan(X_arr).any():
            self.imputer = SimpleImputer(strategy="median")
            X_arr = self.imputer.fit_transform(X_arr)

        self.model = GraphDrone(config)
        self.model.fit(X_arr, y.values.astype(np.float32), expert_specs=specs,
                       problem_type=self.problem_type)
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        preds = self.model.predict(self._to_array(X))
        return pd.Series(preds, index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        # Latest GraphDrone predict() returns probabilities for classification
        probas = self.model.predict(self._to_array(X))
        
        if self.problem_type == "binary":
            # TabArena expects probabilities for all classes
            # probas is likely proba of class 1
            if probas.ndim == 1:
                df_probas = pd.DataFrame({0: 1 - probas, 1: probas}, index=X.index)
            else:
                df_probas = pd.DataFrame(probas, index=X.index)
        else:
            # Multiclass [N, C]
            df_probas = pd.DataFrame(probas, index=X.index)
            
        return df_probas
