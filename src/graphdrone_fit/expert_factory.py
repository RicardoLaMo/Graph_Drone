from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from sklearn.decomposition import PCA

from .portfolio_loader import ConstantPredictor, LinearPredictor, LoadedExpert, LoadedPortfolio
from .view_descriptor import ViewDescriptor


@dataclass(frozen=True)
class ExpertPredictionBatch:
    expert_ids: tuple[str, ...]
    descriptors: tuple[ViewDescriptor, ...]
    predictions: np.ndarray
    full_expert_id: str
    full_index: int
    quality_scores: "np.ndarray | None" = None  # [N, E, 1] variance across bags; None if not bagged


class InputAdapterProtocol(Protocol):
    def fit(self, X: np.ndarray) -> "InputAdapterProtocol":
        ...

    def transform(self, X: np.ndarray) -> np.ndarray:
        ...


@dataclass
class IdentitySelectorAdapter:
    indices: tuple[int, ...]

    def fit(self, X: np.ndarray) -> "IdentitySelectorAdapter":
        matrix = np.asarray(X, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got {matrix.shape}")
        if len(self.indices) == 0:
            raise ValueError("IdentitySelectorAdapter requires at least one feature index")
        if max(self.indices) >= matrix.shape[1]:
            raise ValueError(
                f"IdentitySelectorAdapter expects at least {max(self.indices) + 1} features, got {matrix.shape[1]}"
            )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        matrix = np.asarray(X, dtype=np.float32)
        return matrix[:, self.indices]


@dataclass
class PcaProjectionAdapter:
    n_components: int
    random_state: int = 42
    _pca: PCA | None = field(default=None, init=False, repr=False)

    def fit(self, X: np.ndarray) -> "PcaProjectionAdapter":
        matrix = np.asarray(X, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got {matrix.shape}")
        n_components = min(self.n_components, matrix.shape[0], matrix.shape[1])
        if n_components < 1:
            raise ValueError("PcaProjectionAdapter requires at least one component")
        self._pca = PCA(n_components=n_components, random_state=self.random_state)
        self._pca.fit(matrix)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("PcaProjectionAdapter.fit() must be called before transform()")
        matrix = np.asarray(X, dtype=np.float32)
        return self._pca.transform(matrix).astype(np.float32)


@dataclass(frozen=True)
class ExpertBuildSpec:
    descriptor: ViewDescriptor
    model_kind: str
    input_adapter: InputAdapterProtocol
    model_params: dict[str, object] = field(default_factory=dict)


class BaggedClassifierPredictor:
    """
    Bagged TabPFN: fits ``bag_n`` TabPFNClassifier(n_estimators=nest_each) and
    exposes both mean predictions and per-sample variance as a quality signal.

    Why: a single TabPFNClassifier(n_estimators=8) gives a single prediction per
    expert.  4×(n_estimators=2) with the same total budget lets us estimate
    predictive variance across independently seeded bags — giving the router a
    genuine uncertainty signal rather than always-zero quality tokens.
    """

    def __init__(self, models: list):
        self._models = models

    def _get_bag_preds(self, X: np.ndarray) -> np.ndarray:
        """Stack bag predictions. Shape [B, N, C]. Called once per inference."""
        return np.stack([m.predict_proba(X) for m in self._models], axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Mean probability across bags. Shape [N, C]."""
        return self._get_bag_preds(X).mean(axis=0).astype(np.float32)

    def predict_proba_with_variance(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Single forward pass returning (mean_proba [N,C], variance [N]).

        Variance is mean class-probability variance across bags — high value means
        bags disagree, i.e. the expert is uncertain on this sample.
        """
        preds = self._get_bag_preds(X)  # [B, N, C]
        return preds.mean(axis=0).astype(np.float32), preds.var(axis=0).mean(axis=-1).astype(np.float32)


class PortfolioExpertFactory:
    def __init__(self, portfolio: LoadedPortfolio) -> None:
        self.portfolio = portfolio.validate()
        self.expert_ids = self.portfolio.expert_order
        self.descriptors = self.portfolio.descriptors
        self.full_expert_id = self.portfolio.full_expert_id
        self.full_index = self.expert_ids.index(self.full_expert_id)

    def predict_all(self, X: np.ndarray) -> ExpertPredictionBatch:
        preds: list[np.ndarray] = []
        vars_list: list[np.ndarray] = []
        any_bagged = False

        for expert_id in self.expert_ids:
            expert = self.portfolio.experts[expert_id]
            if isinstance(expert.predictor, BaggedClassifierPredictor):
                any_bagged = True
                # Single forward pass: get mean prediction and variance together,
                # avoiding the double bag-prediction cost of separate calls.
                adapter = expert.input_adapter or (lambda x: x)
                X_view = adapter(X)
                mean_pred, variance = expert.predictor.predict_proba_with_variance(X_view)
                preds.append(mean_pred)
                vars_list.append(variance)
            else:
                preds.append(expert.predict(X))
                vars_list.append(np.zeros(len(X), dtype=np.float32))

        # Regression: each pred is [N] → column_stack → [N, E]
        # Classification: each pred is [N, C] → stack on axis=1 → [N, E, C]
        if preds[0].ndim == 1:
            stacked = np.column_stack(preds).astype(np.float32)   # [N, E]
        else:
            stacked = np.stack(preds, axis=1).astype(np.float32)  # [N, E, C]

        quality_scores = (
            np.stack(vars_list, axis=1)[:, :, np.newaxis].astype(np.float32)
            if any_bagged else None
        )

        return ExpertPredictionBatch(
            expert_ids=self.expert_ids,
            descriptors=self.descriptors,
            predictions=stacked,
            full_expert_id=self.full_expert_id,
            full_index=self.full_index,
            quality_scores=quality_scores,
        )


def fit_portfolio_from_specs(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    specs: tuple[ExpertBuildSpec, ...],
    full_expert_id: str,
    n_jobs: int = -1,
) -> LoadedPortfolio:
    from joblib import Parallel, delayed
    
    matrix = np.asarray(X_train, dtype=np.float32)
    target = np.asarray(y_train).reshape(-1)  # keep dtype; classifiers need int, regressors float
    
    def _fit_single_spec(spec):
        descriptor = spec.descriptor.validate()
        fitted_adapter = spec.input_adapter.fit(matrix)
        X_view = fitted_adapter.transform(matrix)
        predictor = _fit_predictor(
            model_kind=spec.model_kind,
            X_view=X_view,
            y_train=target,
            model_params=spec.model_params,
        )
        return descriptor.expert_id, LoadedExpert(
            descriptor=descriptor,
            predictor=predictor,
            artifact_kind=spec.model_kind,
            input_adapter=fitted_adapter.transform,
        )

    print(f"  -> Fitting {len(specs)} specialists in parallel (n_jobs={n_jobs})...")
    results = Parallel(n_jobs=n_jobs)(delayed(_fit_single_spec)(s) for s in specs)
    
    experts = dict(results)
    expert_order = [s.descriptor.expert_id for s in specs]

    return LoadedPortfolio(
        expert_order=tuple(expert_order),
        experts=experts,
        full_expert_id=full_expert_id,
    ).validate()


def _fit_predictor(
    *,
    model_kind: str,
    X_view: np.ndarray,
    y_train: np.ndarray,
    model_params: dict[str, object],
):
    if model_kind == "constant":
        value = float(model_params.get("value", float(np.mean(y_train))))
        return ConstantPredictor(value=value)

    if model_kind == "linear":
        coef, *_ = np.linalg.lstsq(X_view, y_train, rcond=None)
        return LinearPredictor(coefficients=np.asarray(coef, dtype=np.float32), bias=0.0)

    if model_kind in ("foundation_regressor", "tabpfn_regressor"):
        from tabpfn import TabPFNRegressor

        model = TabPFNRegressor(
            n_estimators=int(model_params.get("n_estimators", 1)),
            random_state=int(model_params.get("random_state", 42)),
            device=model_params.get("device", "auto"),
            ignore_pretraining_limits=bool(model_params.get("ignore_pretraining_limits", len(X_view) > 1000)),
            n_preprocessing_jobs=int(model_params.get("n_preprocessing_jobs", 1)),
        )
        model.fit(X_view, y_train)
        return model

    if model_kind == "foundation_classifier":
        from tabpfn import TabPFNClassifier

        model = TabPFNClassifier(
            n_estimators=int(model_params.get("n_estimators", 1)),
            random_state=int(model_params.get("random_state", 42)),
            device=model_params.get("device", "auto"),
            ignore_pretraining_limits=bool(model_params.get("ignore_pretraining_limits", len(X_view) > 1000)),
            n_preprocessing_jobs=int(model_params.get("n_preprocessing_jobs", 1)),
        )
        model.fit(X_view, y_train)
        return model

    if model_kind == "foundation_classifier_bagged":
        # 4× TabPFNClassifier(n_estimators=2) — same total estimator budget as n_estimators=8
        # but produces predictive variance across independently seeded bags.
        from tabpfn import TabPFNClassifier
        bag_n = int(model_params.get("bag_n", 4))
        nest_each = int(model_params.get("n_estimators", 8)) // bag_n
        nest_each = max(nest_each, 1)
        models = []
        for bag_seed in range(bag_n):
            m = TabPFNClassifier(
                n_estimators=nest_each,
                random_state=int(model_params.get("random_state", 42)) + bag_seed,
                device=model_params.get("device", "auto"),
                ignore_pretraining_limits=bool(model_params.get("ignore_pretraining_limits", len(X_view) > 1000)),
                n_preprocessing_jobs=int(model_params.get("n_preprocessing_jobs", 1)),
            )
            m.fit(X_view, y_train)
            models.append(m)
        return BaggedClassifierPredictor(models)

    raise ValueError(f"Unsupported model_kind={model_kind!r}")
