from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Union

import numpy as np
import torch

from .config import GraphDroneConfig, SetRouterConfig
from .defer_integrator import blend_predictions_torch, integrate_predictions
from .expert_factory import (
    ExpertBuildSpec,
    ExpertPredictionBatch,
    IdentitySelectorAdapter,
    PortfolioExpertFactory,
    fit_portfolio_from_specs,
)
from .geo_ensemble import anchor_geo_poe_blend, learned_geo_poe_blend, learned_geo_poe_blend_torch
from .set_router import (
    LegitimacyGate,
    LegitimacyGateDecision,
    RotorAlignedRouter,
    build_set_router,
)
from .support_encoder import MomentSupportEncoder
from .token_builder import QualityEncoding, UniversalTokenBuilder
from .view_descriptor import ViewDescriptor


def _make_quality_encoding(quality_scores: np.ndarray | None) -> QualityEncoding | None:
    if quality_scores is None:
        return None
    return QualityEncoding(
        tensor=torch.as_tensor(quality_scores),
        feature_names=("bag_variance",),
    )


@dataclass(frozen=True)
class GraphDronePredictResult:
    predictions: np.ndarray
    diagnostics: dict[str, object]
    expert_ids: tuple[str, ...]


def _clf_entropy(predictions: np.ndarray) -> np.ndarray:
    p = np.clip(predictions, 1e-9, 1.0)
    return -np.sum(p * np.log(p), axis=-1).astype(np.float32)


def _coerce_matrix(X: np.ndarray) -> np.ndarray:
    matrix = np.asarray(X, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {matrix.shape}")
    return matrix


def _detect_problem_type(y: np.ndarray, config: GraphDroneConfig) -> tuple[str, int]:
    if config.n_classes > 1:
        return "classification", config.n_classes

    unique_y = np.unique(y)
    is_float_target = np.issubdtype(y.dtype, np.floating) and not np.all(np.mod(y, 1) == 0)
    if not is_float_target and len(unique_y) < 50:
        return "classification", int(unique_y.max() + 1)
    return "regression", 1


def _slice_prediction_batch(batch: ExpertPredictionBatch, mask: np.ndarray) -> ExpertPredictionBatch:
    quality_scores = None if batch.quality_scores is None else batch.quality_scores[mask]
    return ExpertPredictionBatch(
        expert_ids=batch.expert_ids,
        descriptors=batch.descriptors,
        predictions=batch.predictions[mask],
        full_expert_id=batch.full_expert_id,
        full_index=batch.full_index,
        quality_scores=quality_scores,
    )


def _seed_torch_generators(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GraphDrone:
    def __init__(self, config: GraphDroneConfig) -> None:
        self.config = config.validate()
        self._expert_factory: Optional[PortfolioExpertFactory] = None
        self._token_builder = UniversalTokenBuilder(self.config.hyperbolic_descriptors)
        self._support_encoder = MomentSupportEncoder()
        self._router: Optional[torch.nn.Module] = None
        self._legitimacy_gate = LegitimacyGate(self.config.legitimacy_gate)
        self._train_views: dict[str, np.ndarray] = {}
        self._problem_type: str = "regression"
        self._n_classes: int = 1
        self._clf_uses_learned_router: bool = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        expert_specs: Optional[tuple[ExpertBuildSpec, ...]] = None,
        problem_type: Optional[str] = None,
    ) -> "GraphDrone":
        matrix = _coerce_matrix(np.asarray(X, dtype=np.float32))
        self.n_features_in_ = matrix.shape[1]

        if problem_type is not None:
            if problem_type in ("binary", "classification"):
                n_classes = self.config.n_classes if self.config.n_classes > 1 else int(len(np.unique(y)))
                self._problem_type = "classification"
                self._n_classes = n_classes
            else:
                self._problem_type = "regression"
                self._n_classes = 1
        else:
            self._problem_type, self._n_classes = _detect_problem_type(np.asarray(y), self.config)

        is_classification = self._problem_type == "classification"
        is_binary = is_classification and self._n_classes == 2
        y_array = np.asarray(y, dtype=np.int64 if is_classification else np.float32)

        if expert_specs is None:
            expert_specs = self._build_default_specs(matrix, is_classification=is_classification, is_binary=is_binary)

        print(f"  -> Fitting GraphDrone ({self._problem_type}, n_classes={self._n_classes}) on {len(matrix)} samples...")
        self._portfolio = fit_portfolio_from_specs(
            X_train=matrix,
            y_train=y_array,
            specs=expert_specs,
            full_expert_id=self.config.full_expert_id,
        )
        self._expert_factory = PortfolioExpertFactory(self._portfolio)

        for spec in expert_specs:
            fitted_adapter = spec.input_adapter.fit(matrix)
            self._train_views[spec.descriptor.expert_id] = fitted_adapter.transform(matrix)

        if is_classification:
            self._fit_classification_router(matrix, y_array, expert_specs, is_binary=is_binary)
        else:
            self._fit_regression_router(matrix, y_array)
        return self

    def _build_default_specs(
        self,
        matrix: np.ndarray,
        *,
        is_classification: bool,
        is_binary: bool,
    ) -> tuple[ExpertBuildSpec, ...]:
        full_idx = tuple(range(matrix.shape[1]))
        params = {"n_estimators": 8, "device": self.device}
        rng = np.random.RandomState(42)

        if is_classification:
            model_kind = "foundation_classifier_bagged" if is_binary else "foundation_classifier"
            skip_subs = is_binary and (len(matrix) < 500 and matrix.shape[1] < 25)
            full_spec = ExpertBuildSpec(
                descriptor=ViewDescriptor(
                    expert_id=self.config.full_expert_id,
                    family="FULL",
                    view_name="Foundation Full",
                    is_anchor=True,
                    input_dim=matrix.shape[1],
                    input_indices=full_idx,
                ),
                model_kind=model_kind,
                input_adapter=IdentitySelectorAdapter(indices=full_idx),
                model_params=params,
            )
            sub_specs: list[ExpertBuildSpec] = []
            if not skip_subs:
                if is_binary and matrix.shape[1] < 25:
                    sub_specs_config = [(0, 0.5)]
                else:
                    n_features = matrix.shape[1]
                    if n_features <= 10:
                        sub_specs_config = []
                    elif n_features <= 14:
                        sub_specs_config = [(0, 0.6)]
                    else:
                        sub_specs_config = [(0, 0.8), (1, 0.85), (2, 0.9)]

                for sub_seed, sub_frac in sub_specs_config:
                    rng_i = np.random.RandomState(sub_seed)
                    sz_i = max(1, int(matrix.shape[1] * sub_frac))
                    idx_i = tuple(sorted(rng_i.choice(matrix.shape[1], sz_i, replace=False).tolist()))
                    sub_specs.append(
                        ExpertBuildSpec(
                            descriptor=ViewDescriptor(
                                expert_id=f"SUB{sub_seed}",
                                family="structural_subspace",
                                view_name=f"Foundation Sub {sub_seed}",
                                input_dim=sz_i,
                                input_indices=idx_i,
                            ),
                            model_kind=model_kind,
                            input_adapter=IdentitySelectorAdapter(indices=idx_i),
                            model_params=params,
                        )
                    )
            return (full_spec, *sub_specs)

        return (
            ExpertBuildSpec(
                descriptor=ViewDescriptor(
                    expert_id=self.config.full_expert_id,
                    family="FULL",
                    view_name="Foundation Full",
                    is_anchor=True,
                    input_dim=matrix.shape[1],
                    input_indices=full_idx,
                ),
                model_kind="foundation_regressor",
                input_adapter=IdentitySelectorAdapter(indices=full_idx),
                model_params=params,
            ),
        )

    def _seed_router_training(self) -> None:
        _seed_torch_generators(self.config.router.router_seed)

    def _classification_router_config(self, *, is_binary: bool) -> tuple[bool, SetRouterConfig | None]:
        use_learned = is_binary
        if not use_learned:
            return False, None
        if is_binary and self.config.router.kind == "bootstrap_full_only":
            return True, SetRouterConfig(kind="noise_gate_router")
        return True, self.config.router

    def _trainable_params(self) -> list[torch.nn.Parameter]:
        params = [param for param in self._token_builder.trainable_parameters() if param.requires_grad]
        if self._router is not None:
            params.extend(param for param in self._router.parameters() if param.requires_grad)
        return params

    def _post_optimizer_step(self) -> None:
        self._token_builder.project_hyperbolic_parameters_()

    @staticmethod
    def _freeze_module_parameters(module: torch.nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _fit_router_auxiliary_state(
        router: torch.nn.Module,
        aux_tokens: torch.Tensor,
        *,
        full_index: int,
    ) -> None:
        if hasattr(router, "fit_auxiliary_state"):
            router.fit_auxiliary_state(aux_tokens, full_index=full_index)

    @staticmethod
    def _attention_diagnostics(
        *,
        expert_ids: tuple[str, ...],
        full_index: int,
        specialist_weights: torch.Tensor,
    ) -> dict[str, float]:
        weights = specialist_weights.detach().cpu()
        diagnostics: dict[str, float] = {}
        for idx, expert_id in enumerate(expert_ids):
            diagnostics[f"mean_attention_{expert_id}"] = float(weights[:, idx].mean().item())

        if weights.shape[1] <= 1:
            return diagnostics

        non_anchor_mask = torch.ones(weights.shape[1], dtype=torch.bool)
        non_anchor_mask[full_index] = False
        non_anchor = weights[:, non_anchor_mask]
        non_anchor_mass = non_anchor.sum(dim=-1, keepdim=True)
        normalized = non_anchor / non_anchor_mass.clamp(min=1e-9)
        entropy = -(normalized.clamp(min=1e-9) * normalized.clamp(min=1e-9).log()).sum(dim=-1)
        diagnostics["non_anchor_attention_entropy"] = float(entropy.mean().item())
        return diagnostics

    def _optimize_regression_router_module(
        self,
        router: torch.nn.Module,
        *,
        v_tokens_t: torch.Tensor,
        v_preds_t: torch.Tensor,
        y_va_t: torch.Tensor,
        full_index: int,
        anchor_mse_val: float,
        label: str,
    ) -> None:
        import torch.nn.functional as F

        params = [param for param in router.parameters() if param.requires_grad]
        if not params:
            print(f"  -> {label} has no trainable parameters, skipping optimization.")
            return

        optimizer = torch.optim.Adam(params, lr=1e-3)
        best_loss = float("inf")
        patience, wait = 25, 0
        print(
            f"  -> Optimizing {label} on {self.device} "
            f"(Patience={patience}, MSE+ResidualPenalty, anchor_mse={anchor_mse_val:.6f})..."
        )
        autocast_enabled = self.device == "cuda"
        for _ in range(500):
            router.train()
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=autocast_enabled):
                out = router(v_tokens_t, full_index=full_index)
                integ, _, _, _ = blend_predictions_torch(
                    expert_predictions=v_preds_t,
                    specialist_weights=out.specialist_weights,
                    defer_prob=out.defer_prob,
                    full_index=full_index,
                )
                mse = F.mse_loss(integ.squeeze(), y_va_t)
                aux_loss = out.aux_loss if out.aux_loss is not None else mse.new_zeros(())
                loss = mse + 2.0 * F.relu(mse - anchor_mse_val) + aux_loss
            loss.backward()
            optimizer.step()
            self._post_optimizer_step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

    def _sample_aux_rows(self, matrix: np.ndarray, max_rows: int = 1024) -> np.ndarray:
        if len(matrix) <= max_rows:
            return matrix
        rng = np.random.RandomState(42)
        indices = np.sort(rng.choice(len(matrix), size=max_rows, replace=False))
        return matrix[indices]

    def _build_classification_tokens(
        self,
        X: np.ndarray,
        batch: ExpertPredictionBatch,
        *,
        support_train_views: dict[str, np.ndarray] | None = None,
    ):
        support_enc = self._support_encoder.encode(n_rows=len(X), descriptors=batch.descriptors)
        saved_train_views = self._train_views
        if support_train_views is not None:
            self._train_views = support_train_views
        try:
            gora_obs = self._compute_gora_obs(X, batch.descriptors)
        finally:
            self._train_views = saved_train_views
        return self._token_builder.build(
            predictions=_clf_entropy(batch.predictions),
            descriptors=batch.descriptors,
            full_expert_id=batch.full_expert_id,
            support_encoding=support_enc,
            geometric_obs=gora_obs,
            quality_encoding=_make_quality_encoding(batch.quality_scores),
        )

    def _build_regression_tokens(self, X: np.ndarray, batch: ExpertPredictionBatch):
        support_enc = self._support_encoder.encode(n_rows=len(X), descriptors=batch.descriptors)
        gora_obs = self._compute_gora_obs(X, batch.descriptors)
        return self._token_builder.build(
            predictions=batch.predictions,
            descriptors=batch.descriptors,
            full_expert_id=batch.full_expert_id,
            support_encoding=support_enc,
            geometric_obs=gora_obs,
            quality_encoding=_make_quality_encoding(batch.quality_scores),
        )

    def _fit_classification_router(
        self,
        matrix: np.ndarray,
        y: np.ndarray,
        expert_specs: tuple[ExpertBuildSpec, ...],
        *,
        is_binary: bool,
    ) -> None:
        use_learned, router_cfg = self._classification_router_config(is_binary=is_binary)
        self._clf_uses_learned_router = use_learned
        if not use_learned:
            print("  -> Classification (multiclass): using static Geometric PoE blending.")
            return

        from sklearn.model_selection import train_test_split as _tts
        import torch.nn.functional as F

        n_all = len(matrix)
        oof_test_size = 0.25 if n_all <= 1500 else 0.1
        idx_tr90, idx_va = _tts(np.arange(n_all), test_size=oof_test_size, random_state=42, stratify=y)
        X_tr90, X_va = matrix[idx_tr90], matrix[idx_va]
        y_tr90, y_va = y[idx_tr90], y[idx_va]

        oof_specs = tuple(
            ExpertBuildSpec(
                descriptor=spec.descriptor,
                model_kind=spec.model_kind,
                input_adapter=spec.input_adapter,
                model_params={**spec.model_params, "device": "cpu"},
            )
            for spec in expert_specs
        )
        torch.cuda.empty_cache()
        print(f"  -> OOF router training: fitting temporary experts on CPU ({len(X_tr90)} rows)...")
        oof_portfolio = fit_portfolio_from_specs(
            X_train=X_tr90,
            y_train=y_tr90,
            specs=oof_specs,
            full_expert_id=self.config.full_expert_id,
            n_jobs=1,
        )
        oof_factory = PortfolioExpertFactory(oof_portfolio)

        oof_train_views: dict[str, np.ndarray] = {}
        for spec in expert_specs:
            fitted_adapter = spec.input_adapter.fit(X_tr90)
            oof_train_views[spec.descriptor.expert_id] = fitted_adapter.transform(X_tr90)

        va_batch = oof_factory.predict_all(X_va)
        va_tokens = self._build_classification_tokens(X_va, va_batch, support_train_views=oof_train_views)
        if va_tokens.tokens.shape[1] <= 1:
            print("  -> Classification router skipped: anchor-only portfolio leaves nothing to route.")
            self._clf_uses_learned_router = False
            self._router = None
            return
        token_dim = va_tokens.tokens.shape[-1]
        self._seed_router_training()
        self._router = build_set_router(router_cfg, token_dim=token_dim, n_experts=va_tokens.tokens.shape[1]).to(self.device)

        aux_X = self._sample_aux_rows(X_tr90)
        aux_batch = oof_factory.predict_all(aux_X)
        aux_tokens = self._build_classification_tokens(aux_X, aux_batch, support_train_views=oof_train_views)
        if hasattr(self._router, "fit_auxiliary_state"):
            self._router.fit_auxiliary_state(aux_tokens.tokens.to(self.device), full_index=aux_batch.full_index)

        trainable_params = self._trainable_params()
        if not trainable_params:
            print("  -> Classification Router has no trainable parameters, skipping optimization.")
            return

        y_va_t = torch.tensor(y_va, dtype=torch.long, device=self.device)
        log_p = torch.log(torch.tensor(np.clip(va_batch.predictions, 1e-9, 1.0), dtype=torch.float32, device=self.device))
        v_tokens_t = va_tokens.tokens.to(self.device)

        with torch.no_grad():
            log_p_anchor = log_p[:, va_batch.full_index, :]
            anchor_nll_val = F.nll_loss(F.log_softmax(log_p_anchor, dim=-1), y_va_t).item()

        optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
        best_loss = float("inf")
        patience, wait = 25, 0
        print(
            f"  -> Optimizing Classification Router on {self.device} "
            f"(Patience={patience}, NLL+ResidualPenalty, OOF anchor_nll={anchor_nll_val:.4f})..."
        )

        autocast_enabled = self.device == "cuda"
        for _ in range(500):
            self._router.train()
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=autocast_enabled):
                out = self._router(v_tokens_t, full_index=va_batch.full_index)
                log_q = learned_geo_poe_blend_torch(
                    log_p, out.defer_prob, out.specialist_weights, va_batch.full_index
                )
                blend_nll = F.nll_loss(F.log_softmax(log_q, dim=-1), y_va_t)
                aux_loss = out.aux_loss if out.aux_loss is not None else blend_nll.new_zeros(())
                loss = blend_nll + 2.0 * F.relu(blend_nll - anchor_nll_val) + aux_loss
            loss.backward()
            optimizer.step()
            self._post_optimizer_step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        with torch.no_grad():
            out_final = self._router(v_tokens_t, full_index=va_batch.full_index)
            log_q_final = learned_geo_poe_blend_torch(
                log_p, out_final.defer_prob, out_final.specialist_weights, va_batch.full_index
            )
            final_blend_nll = F.nll_loss(F.log_softmax(log_q_final, dim=-1), y_va_t).item()
            mean_defer = float(out_final.defer_prob.mean().item())
        print(
            f"  -> Classification Router trained. "
            f"blend_nll={final_blend_nll:.4f}  anchor_nll={anchor_nll_val:.4f}  mean_defer={mean_defer:.3f}"
        )

    def _fit_regression_router(self, matrix: np.ndarray, y: np.ndarray) -> None:
        from sklearn.model_selection import train_test_split
        import torch.nn.functional as F

        X_tr, X_va, _, y_va = train_test_split(matrix, y, test_size=0.1, random_state=42)
        va_batch = self._expert_factory.predict_all(X_va)
        va_tokens = self._build_regression_tokens(X_va, va_batch)
        token_dim = va_tokens.tokens.shape[-1]
        self._seed_router_training()

        aux_X = self._sample_aux_rows(X_tr)
        aux_batch = self._expert_factory.predict_all(aux_X)
        aux_tokens = self._build_regression_tokens(aux_X, aux_batch)
        aux_tokens_t = aux_tokens.tokens.to(self.device)
        y_va_t = torch.tensor(y_va, dtype=torch.float32, device=self.device)
        v_preds_t = torch.tensor(va_batch.predictions, dtype=torch.float32, device=self.device)
        v_tokens_t = va_tokens.tokens.to(self.device)

        with torch.no_grad():
            anchor_mse_val = F.mse_loss(v_preds_t[:, va_batch.full_index], y_va_t).item()
        if self.config.router.kind == "contextual_transformer_rotor" and self.config.router.freeze_base_router:
            print("  -> Frozen-router rotor ablation: training base router first, then freezing it.")
            base_cfg = replace(
                self.config.router,
                kind="contextual_transformer",
                alignment_lambda=0.0,
                freeze_base_router=False,
            )
            base_router = build_set_router(
                base_cfg,
                token_dim=token_dim,
                n_experts=va_tokens.tokens.shape[1],
            ).to(self.device)
            self._fit_router_auxiliary_state(base_router, aux_tokens_t, full_index=aux_batch.full_index)
            self._optimize_regression_router_module(
                base_router,
                v_tokens_t=v_tokens_t,
                v_preds_t=v_preds_t,
                y_va_t=y_va_t,
                full_index=va_batch.full_index,
                anchor_mse_val=anchor_mse_val,
                label="Frozen-base pre-router",
            )
            self._freeze_module_parameters(base_router)
            self._router = RotorAlignedRouter(
                token_dim=token_dim,
                n_experts=va_tokens.tokens.shape[1],
                base_router=base_router,
                alignment_lambda=self.config.router.alignment_lambda,
                router_kind="contextual_transformer_rotor_frozen_base",
            ).to(self.device)
        else:
            self._router = build_set_router(
                self.config.router,
                token_dim=token_dim,
                n_experts=va_tokens.tokens.shape[1],
            ).to(self.device)

        self._fit_router_auxiliary_state(self._router, aux_tokens_t, full_index=aux_batch.full_index)

        trainable_params = self._trainable_params()
        if not trainable_params:
            print("  -> Router has no trainable parameters (BootstrapFullRouter), skipping optimization.")
            return

        self._optimize_regression_router_module(
            self._router,
            v_tokens_t=v_tokens_t,
            v_preds_t=v_preds_t,
            y_va_t=y_va_t,
            full_index=va_batch.full_index,
            anchor_mse_val=anchor_mse_val,
            label="Router",
        )

    def _compute_gora_obs(self, X: np.ndarray, descriptors: tuple[ViewDescriptor, ...]) -> torch.Tensor:
        from sklearn.neighbors import NearestNeighbors
        from .observers import calculate_kappa, calculate_lid

        all_obs = []
        for descriptor in descriptors:
            X_tr_v = self._train_views[descriptor.expert_id]
            X_v = X[:, list(descriptor.input_indices)] if descriptor.input_indices else X
            knn = NearestNeighbors(n_neighbors=descriptor.preferred_k).fit(X_tr_v)
            dists, indices = knn.kneighbors(X_v)
            kappa = calculate_kappa(X_tr_v, indices).reshape(-1, 1)
            lid = calculate_lid(dists).reshape(-1, 1)
            all_obs.append(np.concatenate([kappa, lid], axis=1))
        return torch.tensor(np.stack(all_obs, axis=1), dtype=torch.float32)

    def _legitimacy_decision(self, batch: ExpertPredictionBatch) -> LegitimacyGateDecision | None:
        gate_config = self.config.legitimacy_gate
        if not gate_config.enabled:
            return None
        if self._problem_type == "regression" and not gate_config.regression_enabled:
            return None
        if self._problem_type == "classification" and self._n_classes == 2 and not gate_config.binary_enabled:
            return None
        if self._problem_type == "classification" and self._n_classes > 2 and not gate_config.multiclass_enabled:
            return None
        anchor_predictions = batch.predictions[:, batch.full_index] if batch.predictions.ndim == 2 else batch.predictions[:, batch.full_index, :]
        return self._legitimacy_gate.evaluate(
            problem_type=self._problem_type,
            anchor_predictions=anchor_predictions,
            expert_predictions=batch.predictions,
            quality_scores=batch.quality_scores,
        )

    def _legitimacy_diagnostics(self, decision: LegitimacyGateDecision | None, *, router_skipped: bool) -> dict[str, object]:
        if decision is None:
            return {
                "early_exit": False,
                "exit_frac": 0.0,
                "legitimacy_metric": "disabled",
                "legitimacy_threshold": float("nan"),
                "legitimacy_score_mean": float("nan"),
                "router_skipped": router_skipped,
            }
        return {
            "early_exit": bool(np.any(decision.exit_mask)),
            "exit_frac": float(np.mean(decision.exit_mask)),
            "legitimacy_metric": decision.metric,
            "legitimacy_threshold": float(decision.threshold),
            "legitimacy_score_mean": float(np.mean(decision.scores)),
            "router_skipped": router_skipped,
        }

    @staticmethod
    def _portfolio_diagnostics(batch: ExpertPredictionBatch) -> dict[str, object]:
        n_experts = len(batch.expert_ids)
        return {
            "n_experts": int(n_experts),
            "n_specialists": int(max(n_experts - 1, 0)),
        }

    def _classification_predictions(
        self,
        matrix: np.ndarray,
        batch: ExpertPredictionBatch,
    ) -> tuple[np.ndarray, dict[str, object]]:
        decision = self._legitimacy_decision(batch)
        anchor_preds = np.asarray(batch.predictions[:, batch.full_index, :], dtype=np.float32)
        gate_mask = decision.exit_mask if decision is not None else np.zeros(len(matrix), dtype=bool)

        if gate_mask.all():
            diagnostics = {
                "router_kind": "legitimacy_gate_anchor_only",
                "mean_defer_prob": 0.0,
            }
            diagnostics.update(self._portfolio_diagnostics(batch))
            diagnostics.update(self._legitimacy_diagnostics(decision, router_skipped=True))
            return anchor_preds, diagnostics

        active_mask = ~gate_mask
        active_batch = batch if active_mask.all() else _slice_prediction_batch(batch, active_mask)
        active_matrix = matrix if active_mask.all() else matrix[active_mask]

        if self._clf_uses_learned_router and self._router is not None:
            tokens = self._build_classification_tokens(active_matrix, active_batch)
            self._router.eval()
            with torch.no_grad():
                token_tensor = tokens.tokens.to(self.device)
                router_out = self._router(token_tensor, full_index=active_batch.full_index)
            active_preds = learned_geo_poe_blend(
                active_batch.predictions,
                router_out.defer_prob,
                router_out.specialist_weights,
                active_batch.full_index,
            )
            diagnostics = {
                "router_kind": router_out.router_kind,
                "mean_defer_prob": float(router_out.defer_prob.mean().item()),
            }
            if router_out.ot_costs is not None:
                diagnostics["mean_ot_cost"] = float(router_out.ot_costs.mean().item())
            if router_out.aux_loss is not None:
                diagnostics["alignment_aux_loss"] = float(router_out.aux_loss.detach().cpu().item())
            if router_out.extra_diagnostics:
                diagnostics.update(router_out.extra_diagnostics)
            diagnostics.update(
                self._attention_diagnostics(
                    expert_ids=active_batch.expert_ids,
                    full_index=active_batch.full_index,
                    specialist_weights=router_out.specialist_weights,
                )
            )
        else:
            active_preds = anchor_geo_poe_blend(
                active_batch.predictions,
                anchor_idx=active_batch.full_index,
                anchor_weight=5.0,
            )
            diagnostics = {
                "router_kind": "geo_poe",
                "mean_defer_prob": float("nan"),
            }
            diagnostics.update(self._portfolio_diagnostics(active_batch))

        preds = anchor_preds.copy()
        preds[active_mask] = np.asarray(active_preds, dtype=np.float32)
        diagnostics.update(self._legitimacy_diagnostics(decision, router_skipped=False))
        return preds.astype(np.float32), diagnostics

    def _regression_predictions(
        self,
        matrix: np.ndarray,
        batch: ExpertPredictionBatch,
    ) -> tuple[np.ndarray, dict[str, object]]:
        decision = self._legitimacy_decision(batch)
        anchor_preds = np.asarray(batch.predictions[:, batch.full_index], dtype=np.float32)
        gate_mask = decision.exit_mask if decision is not None else np.zeros(len(matrix), dtype=bool)

        if gate_mask.all():
            diagnostics = {
                "router_kind": "legitimacy_gate_anchor_only",
                "mean_defer_prob": 0.0,
                "full_index": int(batch.full_index),
            }
            diagnostics.update(self._portfolio_diagnostics(batch))
            diagnostics.update(self._legitimacy_diagnostics(decision, router_skipped=True))
            return anchor_preds, diagnostics

        active_mask = ~gate_mask
        active_batch = batch if active_mask.all() else _slice_prediction_batch(batch, active_mask)
        active_matrix = matrix if active_mask.all() else matrix[active_mask]

        tokens = self._build_regression_tokens(active_matrix, active_batch)
        self._router.eval()
        with torch.no_grad():
            token_tensor = tokens.tokens.to(self.device)
            router_out = self._router(token_tensor, full_index=active_batch.full_index)
            integration = integrate_predictions(expert_predictions=active_batch.predictions, router_outputs=router_out)

        preds = anchor_preds.copy()
        preds[active_mask] = integration.predictions
        diagnostics = dict(integration.diagnostics)
        if router_out.ot_costs is not None:
            diagnostics["mean_ot_cost"] = float(router_out.ot_costs.mean().item())
        if router_out.aux_loss is not None:
            diagnostics["alignment_aux_loss"] = float(router_out.aux_loss.detach().cpu().item())
        if router_out.extra_diagnostics:
            diagnostics.update(router_out.extra_diagnostics)
        diagnostics.update(
            self._attention_diagnostics(
                expert_ids=active_batch.expert_ids,
                full_index=active_batch.full_index,
                specialist_weights=router_out.specialist_weights,
            )
        )
        diagnostics.update(self._legitimacy_diagnostics(decision, router_skipped=False))
        return preds.astype(np.float32), diagnostics

    def predict(self, X: np.ndarray, return_diagnostics: bool = False) -> Union[np.ndarray, GraphDronePredictResult]:
        matrix = _coerce_matrix(np.asarray(X, dtype=np.float32))
        batch = self._expert_factory.predict_all(matrix)

        if self._problem_type == "classification":
            preds, diagnostics = self._classification_predictions(matrix, batch)
        else:
            preds, diagnostics = self._regression_predictions(matrix, batch)

        if return_diagnostics:
            return GraphDronePredictResult(
                predictions=preds,
                expert_ids=batch.expert_ids,
                diagnostics=diagnostics,
            )
        return preds
