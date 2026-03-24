from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Union

import numpy as np
import torch

from .config import GraphDroneConfig, SetRouterConfig
from .defer_integrator import blend_predictions_torch, integrate_predictions, normalize_non_anchor_weights
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
    TaskConditionedRouter,
    build_set_router,
)
from .support_encoder import MomentSupportEncoder
from .task_conditioned_prior import build_task_context_frame_from_router_tokens, compute_task_prior_from_bank
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
        self._router_fit_diagnostics: dict[str, float] = {}
        self._task_prior_diagnostics: dict[str, object] = {}
        self._router_training_force_anchor_only: bool = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _record_regression_router_fallback(
        self,
        *,
        stage: str,
        reason: str,
        nonfinite: bool,
        **extra: object,
    ) -> None:
        self._router_fit_diagnostics["regression_router_fallback_stage"] = stage
        self._router_fit_diagnostics["regression_router_fallback_reason"] = reason
        self._router_fit_diagnostics["validation_router_training_nonfinite_flag"] = float(nonfinite)
        for key, value in extra.items():
            self._router_fit_diagnostics[key] = value

    @staticmethod
    def _regression_fallback_defaults() -> dict[str, object]:
        return {
            "router_nonfinite_fallback": False,
            "regression_router_fallback_stage": "none",
            "regression_router_fallback_reason": "none",
            "validation_router_tokens_finite_flag": float("nan"),
            "validation_router_predictions_finite_flag": float("nan"),
            "validation_router_targets_finite_flag": float("nan"),
            "validation_anchor_mse_finite_flag": float("nan"),
            "prediction_router_tokens_finite_flag": float("nan"),
            "prediction_router_weights_finite_flag": float("nan"),
            "prediction_router_defer_finite_flag": float("nan"),
        }

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
        self._router_fit_diagnostics = {}
        self._task_prior_diagnostics = {}
        self._router_training_force_anchor_only = False
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
            subspace_indices = (
                self._binary_classification_subspaces(matrix.shape[1])
                if is_binary
                else self._multiclass_subspaces(matrix.shape[1])
            )
            for sub_idx, idx_i in enumerate(subspace_indices):
                sub_specs.append(
                    ExpertBuildSpec(
                        descriptor=ViewDescriptor(
                            expert_id=f"SUB{sub_idx}",
                            family="structural_subspace",
                            view_name=f"Foundation Sub {sub_idx}",
                            input_dim=len(idx_i),
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

    @staticmethod
    def _multiclass_subspaces(n_features: int) -> tuple[tuple[int, ...], ...]:
        if n_features <= 10:
            return ()
        if n_features <= 14:
            rng_i = np.random.RandomState(0)
            sz_i = max(1, int(n_features * 0.6))
            idx_i = tuple(sorted(rng_i.choice(n_features, sz_i, replace=False).tolist()))
            return (idx_i,)
        configs = []
        for sub_seed, sub_frac in [(0, 0.8), (1, 0.85), (2, 0.9)]:
            rng_i = np.random.RandomState(sub_seed)
            sz_i = max(1, int(n_features * sub_frac))
            idx_i = tuple(sorted(rng_i.choice(n_features, sz_i, replace=False).tolist()))
            configs.append(idx_i)
        return tuple(configs)

    @staticmethod
    def _binary_classification_subspaces(n_features: int) -> tuple[tuple[int, ...], ...]:
        if n_features <= 2:
            return ()
        all_idx = np.arange(n_features)
        split = max(1, n_features // 2)
        lower = tuple(all_idx[:split].tolist())
        upper = tuple(all_idx[-split:].tolist())
        views: list[tuple[int, ...]] = []
        for idx in (lower, upper):
            if len(idx) < n_features and idx not in views:
                views.append(idx)
        for seed, frac in [(0, 0.75), (1, 0.6)]:
            rng = np.random.RandomState(seed)
            sz = min(n_features - 1, max(2, int(round(n_features * frac))))
            idx = tuple(sorted(rng.choice(n_features, sz, replace=False).tolist()))
            if len(idx) < n_features and idx not in views:
                views.append(idx)
        return tuple(views)

    def _seed_router_training(self) -> None:
        _seed_torch_generators(self.config.router.router_seed)

    def _classification_router_config(self, *, is_binary: bool) -> tuple[bool, SetRouterConfig | None]:
        use_learned = is_binary
        if not use_learned:
            return False, None
        if is_binary and self.config.router.kind == "bootstrap_full_only":
            return True, replace(self.config.router, kind="noise_gate_router")
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

    @staticmethod
    def _regression_residual_usefulness_diagnostics(
        *,
        expert_predictions: torch.Tensor,
        y_true: torch.Tensor,
        specialist_weights: torch.Tensor,
        defer_prob: torch.Tensor,
        full_index: int,
    ) -> dict[str, float]:
        stats = GraphDrone._regression_residual_usefulness_tensors(
            expert_predictions=expert_predictions,
            y_true=y_true,
            specialist_weights=specialist_weights,
            defer_prob=defer_prob,
            full_index=full_index,
        )
        active = stats["active_mask"]
        if not bool(active.any().item()):
            return {}
        return {
            "validation_best_specialist_advantage_score": float(stats["best_advantage"][active].mean().item()),
            "validation_weighted_specialist_advantage_score": float(
                stats["weighted_advantage"][active].mean().item()
            ),
            "validation_defer_weighted_specialist_advantage_score": float(
                stats["realized_advantage"][active].mean().item()
            ),
            "validation_top_specialist_advantage_score": float(stats["top_advantage"][active].mean().item()),
            "validation_positive_specialist_mass": float(stats["positive_mass"][active].mean().item()),
            "validation_top_specialist_positive_rate": float(
                (stats["top_advantage"][active] > 0).float().mean().item()
            ),
            "validation_positive_specialist_opportunity_score": float(
                stats["positive_opportunity"][active].mean().item()
            ),
            "validation_residual_usefulness_gap": float(
                stats["residual_usefulness_gap"][active].mean().item()
            ),
            "validation_allocation_usefulness_score": float(
                GraphDrone._regression_allocation_usefulness_from_stats(stats).item()
            ),
            "validation_robust_allocation_usefulness_score": float(
                GraphDrone._regression_robust_allocation_usefulness_from_stats(stats).item()
            ),
            "validation_conservative_allocation_penalty": float(
                GraphDrone._regression_conservative_allocation_penalty_from_stats(stats).item()
            ),
            "validation_conservative_allocation_active_rate": float(
                GraphDrone._regression_conservative_allocation_active_rate_from_stats(stats).item()
            ),
        }

    @staticmethod
    def _regression_allocation_usefulness_from_stats(stats: dict[str, torch.Tensor]) -> torch.Tensor:
        active = stats["active_mask"]
        if not bool(active.any().item()):
            reference = stats["weighted_advantage"]
            return reference.new_zeros(())
        weighted_advantage = stats["weighted_advantage"][active].mean()
        positive_mass = stats["positive_mass"][active].mean()
        return weighted_advantage + 0.5 * positive_mass

    @staticmethod
    def _regression_allocation_usefulness_from_mask(
        stats: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(mask.any().item()):
            reference = stats["weighted_advantage"]
            return reference.new_zeros(())
        weighted_advantage = stats["weighted_advantage"][mask].mean()
        positive_mass = stats["positive_mass"][mask].mean()
        return weighted_advantage + 0.5 * positive_mass

    @staticmethod
    def _regression_robust_allocation_usefulness_from_stats(stats: dict[str, torch.Tensor]) -> torch.Tensor:
        active = stats["active_mask"]
        if not bool(active.any().item()):
            reference = stats["weighted_advantage"]
            return reference.new_zeros(())
        indices = torch.arange(active.shape[0], device=active.device)
        even_mask = active & ((indices % 2) == 0)
        odd_mask = active & ((indices % 2) == 1)
        even_score = GraphDrone._regression_allocation_usefulness_from_mask(stats, even_mask)
        odd_score = GraphDrone._regression_allocation_usefulness_from_mask(stats, odd_mask)
        return torch.minimum(even_score, odd_score)

    @staticmethod
    def _regression_conservative_allocation_active_rate_from_stats(
        stats: dict[str, torch.Tensor],
        *,
        opportunity_threshold: float = 0.1,
    ) -> torch.Tensor:
        active = stats["active_mask"]
        if not bool(active.any().item()):
            reference = stats["weighted_advantage"]
            return reference.new_zeros(())
        confident = active & (stats["best_advantage"] > opportunity_threshold)
        return confident.float().mean()

    @staticmethod
    def _regression_conservative_allocation_penalty_from_stats(
        stats: dict[str, torch.Tensor],
        *,
        opportunity_threshold: float = 0.1,
    ) -> torch.Tensor:
        active = stats["active_mask"]
        if not bool(active.any().item()):
            reference = stats["weighted_advantage"]
            return reference.new_zeros(())
        confident = active & (stats["best_advantage"] > opportunity_threshold)
        if not bool(confident.any().item()):
            reference = stats["weighted_advantage"]
            return reference.new_zeros(())
        confidence = torch.relu(stats["best_advantage"][confident] - opportunity_threshold)
        negative_mass = 1.0 - stats["positive_mass"][confident]
        return (confidence * negative_mass).mean()

    @staticmethod
    def _regression_residual_usefulness_tensors(
        *,
        expert_predictions: torch.Tensor,
        y_true: torch.Tensor,
        specialist_weights: torch.Tensor,
        defer_prob: torch.Tensor,
        full_index: int,
    ) -> dict[str, torch.Tensor]:
        if expert_predictions.ndim != 2:
            raise ValueError(f"Expected [N, E] expert predictions, got shape {tuple(expert_predictions.shape)}")

        non_anchor_mask = torch.ones(expert_predictions.shape[1], dtype=torch.bool, device=expert_predictions.device)
        non_anchor_mask[full_index] = False
        if int(non_anchor_mask.sum().item()) == 0:
            zeros = torch.zeros(
                expert_predictions.shape[0],
                dtype=expert_predictions.dtype,
                device=expert_predictions.device,
            )
            return {
                "active_mask": torch.zeros(
                    expert_predictions.shape[0],
                    dtype=torch.bool,
                    device=expert_predictions.device,
                ),
                "weighted_advantage": zeros,
                "best_advantage": zeros,
                "realized_advantage": zeros,
                "top_advantage": zeros,
                "positive_mass": zeros,
                "positive_opportunity": zeros,
                "residual_usefulness_gap": zeros,
            }

        anchor_pred = expert_predictions[:, full_index]
        anchor_abs_err = (y_true - anchor_pred).abs().unsqueeze(1)
        specialist_abs_err = (y_true.unsqueeze(1) - expert_predictions[:, non_anchor_mask]).abs()
        denom = anchor_abs_err + specialist_abs_err + 1e-6
        advantage = (anchor_abs_err - specialist_abs_err) / denom

        normalized_weights, specialist_mass = normalize_non_anchor_weights(
            specialist_weights,
            full_index=full_index,
        )
        specialist_weights_only = normalized_weights[:, non_anchor_mask]
        has_specialist_mass = specialist_mass.squeeze(1) > 0
        effective_defer = torch.where(
            specialist_mass > 0,
            defer_prob,
            torch.zeros_like(defer_prob),
        ).squeeze(1)
        weighted_advantage = (specialist_weights_only * advantage).sum(dim=1)
        best_advantage = advantage.max(dim=1).values
        positive_mass = (specialist_weights_only * (advantage > 0).float()).sum(dim=1)
        top_indices = specialist_weights_only.argmax(dim=1, keepdim=True)
        top_advantage = advantage.gather(1, top_indices).squeeze(1)
        realized_advantage = effective_defer * weighted_advantage
        positive_opportunity = torch.relu(best_advantage)
        residual_usefulness_gap = torch.relu(positive_opportunity - realized_advantage)
        return {
            "active_mask": has_specialist_mass,
            "weighted_advantage": weighted_advantage,
            "best_advantage": best_advantage,
            "realized_advantage": realized_advantage,
            "top_advantage": top_advantage,
            "positive_mass": positive_mass,
            "positive_opportunity": positive_opportunity,
            "residual_usefulness_gap": residual_usefulness_gap,
        }

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
            self._record_regression_router_fallback(
                stage="train_setup",
                reason="no_trainable_params",
                nonfinite=False,
            )
            return

        optimizer = torch.optim.Adam(params, lr=1e-3)
        best_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in router.state_dict().items()
        }
        best_loss = float("inf")
        patience, wait = 25, 0
        saw_nonfinite = False
        print(
            f"  -> Optimizing {label} on {self.device} "
            f"(Patience={patience}, MSE+ResidualPenalty"
            f"{'+AllocationUsefulness' if self.config.router.allocation_usefulness_lambda > 0 else ''}"
            f"{'+RobustAllocation' if self.config.router.robust_allocation_usefulness_lambda > 0 else ''}"
            f"{'+ConservativeAllocation' if self.config.router.conservative_allocation_lambda > 0 else ''}"
            f"{'+UsefulnessGap' if self.config.router.residual_usefulness_lambda > 0 else ''}, "
            f"anchor_mse={anchor_mse_val:.6f})..."
        )
        autocast_enabled = self.device == "cuda"
        self._router_fit_diagnostics["validation_router_tokens_finite_flag"] = float(torch.isfinite(v_tokens_t).all().item())
        self._router_fit_diagnostics["validation_router_predictions_finite_flag"] = float(
            torch.isfinite(v_preds_t).all().item()
        )
        self._router_fit_diagnostics["validation_router_targets_finite_flag"] = float(torch.isfinite(y_va_t).all().item())
        self._router_fit_diagnostics["validation_anchor_mse_finite_flag"] = float(np.isfinite(anchor_mse_val))
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
                if self.config.router.residual_usefulness_lambda > 0:
                    usefulness = self._regression_residual_usefulness_tensors(
                        expert_predictions=v_preds_t,
                        y_true=y_va_t,
                        specialist_weights=out.specialist_weights,
                        defer_prob=out.defer_prob,
                        full_index=full_index,
                    )
                    active = usefulness["active_mask"]
                    if bool(active.any().item()):
                        loss = (
                            loss
                            + self.config.router.residual_usefulness_lambda
                            * usefulness["residual_usefulness_gap"][active].mean()
                        )
                if self.config.router.allocation_usefulness_lambda > 0:
                    usefulness = self._regression_residual_usefulness_tensors(
                        expert_predictions=v_preds_t,
                        y_true=y_va_t,
                        specialist_weights=out.specialist_weights,
                        defer_prob=out.defer_prob,
                        full_index=full_index,
                    )
                    allocation_score = self._regression_allocation_usefulness_from_stats(usefulness)
                    loss = loss - self.config.router.allocation_usefulness_lambda * allocation_score
                if self.config.router.robust_allocation_usefulness_lambda > 0:
                    usefulness = self._regression_residual_usefulness_tensors(
                        expert_predictions=v_preds_t,
                        y_true=y_va_t,
                        specialist_weights=out.specialist_weights,
                        defer_prob=out.defer_prob,
                        full_index=full_index,
                    )
                    robust_allocation_score = self._regression_robust_allocation_usefulness_from_stats(usefulness)
                    loss = loss - self.config.router.robust_allocation_usefulness_lambda * robust_allocation_score
                if self.config.router.conservative_allocation_lambda > 0:
                    usefulness = self._regression_residual_usefulness_tensors(
                        expert_predictions=v_preds_t,
                        y_true=y_va_t,
                        specialist_weights=out.specialist_weights,
                        defer_prob=out.defer_prob,
                        full_index=full_index,
                    )
                    conservative_penalty = self._regression_conservative_allocation_penalty_from_stats(
                        usefulness,
                        opportunity_threshold=self.config.router.conservative_allocation_opportunity_threshold,
                    )
                    loss = loss + self.config.router.conservative_allocation_lambda * conservative_penalty
            if not torch.isfinite(loss):
                saw_nonfinite = True
                print(f"  -> {label} produced non-finite loss; restoring best router state.")
                self._record_regression_router_fallback(
                    stage="train_loss",
                    reason="nonfinite_loss",
                    nonfinite=True,
                )
                break
            loss.backward()
            grad_is_finite = True
            for param in params:
                if param.grad is None:
                    continue
                if not torch.isfinite(param.grad).all():
                    grad_is_finite = False
                    break
            if not grad_is_finite:
                saw_nonfinite = True
                print(f"  -> {label} produced non-finite gradients; restoring best router state.")
                optimizer.zero_grad(set_to_none=True)
                self._record_regression_router_fallback(
                    stage="train_gradients",
                    reason="nonfinite_gradients",
                    nonfinite=True,
                )
                break
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            self._post_optimizer_step()
            params_are_finite = all(torch.isfinite(param).all() for param in params)
            if not params_are_finite:
                saw_nonfinite = True
                print(f"  -> {label} produced non-finite parameters; restoring best router state.")
                self._record_regression_router_fallback(
                    stage="train_parameters",
                    reason="nonfinite_parameters",
                    nonfinite=True,
                )
                break
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in router.state_dict().items()
                }
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
        router.load_state_dict(best_state)
        self._router_fit_diagnostics["validation_router_training_nonfinite_flag"] = float(saw_nonfinite)
        self._router_fit_diagnostics["validation_router_best_objective"] = (
            float(best_loss) if np.isfinite(best_loss) else float("nan")
        )
        if saw_nonfinite:
            self._router_training_force_anchor_only = True

    @staticmethod
    def _normalize_regression_router_targets(
        *,
        expert_predictions: torch.Tensor,
        y_true: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        y_center = torch.median(y_true)
        centered_targets = y_true - y_center
        target_scale = centered_targets.std(unbiased=False)
        if not torch.isfinite(target_scale) or float(target_scale.item()) < 1e-6:
            target_scale = centered_targets.abs().mean()
        if not torch.isfinite(target_scale) or float(target_scale.item()) < 1e-6:
            target_scale = y_true.new_tensor(1.0)
        normalized_predictions = (expert_predictions - y_center) / target_scale
        normalized_targets = centered_targets / target_scale
        diagnostics = {
            "validation_router_target_center": float(y_center.detach().cpu().item()),
            "validation_router_target_scale": float(target_scale.detach().cpu().item()),
        }
        return normalized_predictions, normalized_targets, diagnostics

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

    @staticmethod
    def _normalize_regression_token_predictions(
        predictions: np.ndarray,
        *,
        full_index: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        pred = np.asarray(predictions, dtype=np.float32)
        anchor = pred[:, full_index]
        center = float(np.median(anchor))
        scale = float(np.std(anchor - center))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = float(np.mean(np.abs(anchor - center)))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = float(np.std(pred))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        normalized = (pred - center) / scale
        diagnostics = {
            "regression_token_prediction_center": center,
            "regression_token_prediction_scale": scale,
        }
        return normalized.astype(np.float32), diagnostics

    def _build_regression_tokens(self, X: np.ndarray, batch: ExpertPredictionBatch):
        support_enc = self._support_encoder.encode(n_rows=len(X), descriptors=batch.descriptors)
        gora_obs = self._compute_gora_obs(X, batch.descriptors)
        normalized_predictions, token_norm_diagnostics = self._normalize_regression_token_predictions(
            batch.predictions,
            full_index=batch.full_index,
        )
        self._router_fit_diagnostics.update(token_norm_diagnostics)
        return self._token_builder.build(
            predictions=normalized_predictions,
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
        self._router = self._maybe_attach_task_prior_router(
            router=self._router,
            router_cfg=router_cfg,
            tokens=va_tokens.tokens,
            descriptors=va_batch.descriptors,
            task_type="classification",
        )

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
        task_prior_penalty_lambda = 0.0 if router_cfg is None else float(router_cfg.task_prior_defer_penalty_lambda)
        task_prior_defer_target = 0.8 if router_cfg is None else float(router_cfg.task_prior_defer_target)
        task_prior_rank_loss_lambda = 0.0 if router_cfg is None else float(router_cfg.task_prior_rank_loss_lambda)
        task_prior_rank_margin = 0.0 if router_cfg is None else float(router_cfg.task_prior_rank_margin)
        task_prior_confidence = self._task_prior_confidence_scale()
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
                if task_prior_penalty_lambda > 0 and task_prior_confidence > 0:
                    mean_defer = out.defer_prob.mean()
                    defer_penalty = (mean_defer - task_prior_defer_target) ** 2
                    loss = loss + task_prior_penalty_lambda * task_prior_confidence * defer_penalty
                if task_prior_rank_loss_lambda > 0 and task_prior_confidence > 0 and log_q.shape[-1] == 2:
                    pos_score = log_q[:, 1] - log_q[:, 0]
                    rank_loss = self._binary_pairwise_rank_loss(
                        pos_score,
                        y_va_t,
                        margin=task_prior_rank_margin,
                    )
                    loss = loss + task_prior_rank_loss_lambda * task_prior_confidence * rank_loss
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

    def _task_prior_confidence_scale(self) -> float:
        if not self._task_prior_diagnostics:
            return 0.0
        if not bool(self._task_prior_diagnostics.get("task_prior_feedback_used", False)):
            return 0.0
        exact = 1.0 if bool(self._task_prior_diagnostics.get("task_prior_exact_reuse_used", False)) else 0.0
        top_prob = float(self._task_prior_diagnostics.get("task_prior_top_neighbor_prob", 0.0) or 0.0)
        entropy = float(self._task_prior_diagnostics.get("task_prior_entropy", 0.0) or 0.0)
        entropy_scale = max(0.0, 1.0 - entropy / 2.0)
        return max(exact, min(1.0, top_prob + entropy_scale))

    @staticmethod
    def _binary_pairwise_rank_loss(
        pos_score: torch.Tensor,
        y_true: torch.Tensor,
        *,
        margin: float = 0.0,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        pos_mask = y_true == 1
        neg_mask = y_true == 0
        if not bool(pos_mask.any()) or not bool(neg_mask.any()):
            return pos_score.new_zeros(())
        pos_scores = pos_score[pos_mask]
        neg_scores = pos_score[neg_mask]
        pairwise_margin = pos_scores[:, None] - neg_scores[None, :] - margin
        return F.softplus(-pairwise_margin).mean()

    def _maybe_attach_task_prior_router(
        self,
        *,
        router: torch.nn.Module,
        router_cfg: SetRouterConfig | None,
        tokens: torch.Tensor,
        descriptors: tuple[ViewDescriptor, ...],
        task_type: str,
    ) -> torch.nn.Module:
        if router_cfg is None or router_cfg.task_prior_bank_dir is None:
            self._task_prior_diagnostics = {}
            return router
        task_context_df = build_task_context_frame_from_router_tokens(
            tokens=tokens,
            descriptors=descriptors,
            dataset_name=router_cfg.task_prior_dataset_key or "__current__",
            task_type=task_type,
        )
        query_dataset = router_cfg.task_prior_dataset_key or "__current__"
        prior_bundle = compute_task_prior_from_bank(
            bank_dir=router_cfg.task_prior_bank_dir,
            task_context_df=task_context_df,
            encoder_kind=router_cfg.task_prior_encoder_kind,
            device=self.device,
            query_dataset=query_dataset,
            top_k=3,
            exact_reuse_blend=router_cfg.task_prior_exact_reuse_blend,
        )
        conditioned = TaskConditionedRouter(
            token_dim=tokens.shape[-1],
            prior_dim=prior_bundle["prior_vector"].shape[0],
            base_router=router,
            strength=router_cfg.task_prior_strength,
            mode=router_cfg.task_prior_mode,
            local_gate_alpha=router_cfg.task_prior_local_gate_alpha,
            expert_local_gate_alpha=router_cfg.task_prior_expert_local_gate_alpha,
            router_kind=f"{getattr(router, 'router_kind', router_cfg.kind)}_task_prior",
        ).to(self.device)
        conditioned.set_task_prior_context(prior_bundle["prior_vector"].to(self.device))
        query_result = prior_bundle["query_result"]
        top_neighbors = query_result.get("top_neighbors", [])
        base_neighbor_probs = query_result.get("base_neighbor_probabilities", {})
        base_top_neighbor = max(base_neighbor_probs, key=base_neighbor_probs.get) if base_neighbor_probs else ""
        self._task_prior_diagnostics = {
            "task_prior_training_objective": str(prior_bundle["training_objective"]),
            "task_prior_encoder_kind": str(prior_bundle["encoder_kind"]),
            "task_prior_top_neighbor": top_neighbors[0]["dataset"] if top_neighbors else "",
            "task_prior_top_neighbor_prob": float(top_neighbors[0]["probability"]) if top_neighbors else float("nan"),
            "task_prior_base_top_neighbor": base_top_neighbor,
            "task_prior_base_top_neighbor_prob": float(base_neighbor_probs.get(base_top_neighbor, float("nan"))),
            "task_prior_entropy": float(query_result.get("soft_neighbor_entropy", float("nan"))),
            "task_prior_exact_reuse_available": bool(query_result.get("exact_reuse_available", False)),
            "task_prior_exact_reuse_blend": float(prior_bundle.get("exact_reuse_blend", 0.0)),
            "task_prior_exact_reuse_used": bool(prior_bundle.get("exact_reuse_used", False)),
            "task_prior_mode": str(router_cfg.task_prior_mode),
            "task_prior_local_gate_alpha": float(router_cfg.task_prior_local_gate_alpha),
            "task_prior_expert_local_gate_alpha": float(router_cfg.task_prior_expert_local_gate_alpha),
            "task_prior_query_dataset": query_dataset,
            "task_prior_feedback_used": bool(query_result.get("feedback_used", False)),
            "task_prior_feedback_top_source": (
                query_result.get("top_source_datasets", [{}])[0].get("dataset", "")
                if query_result.get("top_source_datasets")
                else ""
            ),
            "task_prior_feedback_top_source_weight": (
                float(query_result.get("top_source_datasets", [{}])[0].get("weight", float("nan")))
                if query_result.get("top_source_datasets")
                else float("nan")
            ),
        }
        return conditioned

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
        v_preds_t_norm, y_va_t_norm, normalization_diagnostics = self._normalize_regression_router_targets(
            expert_predictions=v_preds_t,
            y_true=y_va_t,
        )
        self._router_fit_diagnostics.update(normalization_diagnostics)
        v_tokens_t = va_tokens.tokens.to(self.device)

        with torch.no_grad():
            anchor_mse_val = F.mse_loss(v_preds_t_norm[:, va_batch.full_index], y_va_t_norm).item()
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
                v_preds_t=v_preds_t_norm,
                y_va_t=y_va_t_norm,
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
        self._router = self._maybe_attach_task_prior_router(
            router=self._router,
            router_cfg=self.config.router,
            tokens=va_tokens.tokens,
            descriptors=va_batch.descriptors,
            task_type="regression",
        )

        self._fit_router_auxiliary_state(self._router, aux_tokens_t, full_index=aux_batch.full_index)

        trainable_params = self._trainable_params()
        if not trainable_params:
            print("  -> Router has no trainable parameters (BootstrapFullRouter), skipping optimization.")
            return

        self._optimize_regression_router_module(
            self._router,
            v_tokens_t=v_tokens_t,
            v_preds_t=v_preds_t_norm,
            y_va_t=y_va_t_norm,
            full_index=va_batch.full_index,
            anchor_mse_val=anchor_mse_val,
            label="Router",
        )
        self._router.eval()
        with torch.no_grad():
            out_final = self._router(v_tokens_t, full_index=va_batch.full_index)
            usefulness_diagnostics = self._regression_residual_usefulness_diagnostics(
                expert_predictions=v_preds_t,
                y_true=y_va_t,
                specialist_weights=out_final.specialist_weights,
                defer_prob=out_final.defer_prob,
                full_index=va_batch.full_index,
            )
        if usefulness_diagnostics:
            self._router_fit_diagnostics.update(usefulness_diagnostics)
            self._router_fit_diagnostics["validation_residual_usefulness_lambda"] = float(
                self.config.router.residual_usefulness_lambda
            )
            self._router_fit_diagnostics["validation_allocation_usefulness_lambda"] = float(
                self.config.router.allocation_usefulness_lambda
            )
            self._router_fit_diagnostics["validation_robust_allocation_usefulness_lambda"] = float(
                self.config.router.robust_allocation_usefulness_lambda
            )
            self._router_fit_diagnostics["validation_conservative_allocation_lambda"] = float(
                self.config.router.conservative_allocation_lambda
            )
            print(
                "  -> Regression router usefulness: "
                f"weighted_adv={usefulness_diagnostics['validation_weighted_specialist_advantage_score']:.4f} "
                f"best_adv={usefulness_diagnostics['validation_best_specialist_advantage_score']:.4f} "
                f"gap={usefulness_diagnostics['validation_residual_usefulness_gap']:.4f} "
                f"positive_mass={usefulness_diagnostics['validation_positive_specialist_mass']:.4f} "
                f"allocation={usefulness_diagnostics['validation_allocation_usefulness_score']:.4f} "
                f"robust_allocation={usefulness_diagnostics['validation_robust_allocation_usefulness_score']:.4f} "
                f"conservative_penalty={usefulness_diagnostics['validation_conservative_allocation_penalty']:.4f} "
                f"conservative_active={usefulness_diagnostics['validation_conservative_allocation_active_rate']:.4f}"
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
            diagnostics.update(self._task_prior_diagnostics)
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
            diagnostics.update(self._regression_fallback_defaults())
            diagnostics.update(self._portfolio_diagnostics(batch))
            diagnostics.update(self._router_fit_diagnostics)
            diagnostics.update(self._legitimacy_diagnostics(decision, router_skipped=True))
            return anchor_preds, diagnostics

        active_mask = ~gate_mask
        active_batch = batch if active_mask.all() else _slice_prediction_batch(batch, active_mask)
        active_matrix = matrix if active_mask.all() else matrix[active_mask]

        if self._router_training_force_anchor_only:
            preds = anchor_preds.copy()
            preds[active_mask] = np.asarray(active_batch.predictions[:, active_batch.full_index], dtype=np.float32)
            diagnostics = {
                "router_kind": "router_training_nonfinite_anchor_only",
                "mean_defer_prob": 0.0,
                "effective_defer_rate": 0.0,
                "full_index": int(active_batch.full_index),
                "router_nonfinite_fallback": True,
                "regression_router_fallback_stage": self._router_fit_diagnostics.get(
                    "regression_router_fallback_stage", "train_unknown"
                ),
                "regression_router_fallback_reason": self._router_fit_diagnostics.get(
                    "regression_router_fallback_reason", "unknown"
                ),
            }
            base_defaults = self._regression_fallback_defaults()
            base_defaults.update(diagnostics)
            diagnostics = base_defaults
            diagnostics.update(self._portfolio_diagnostics(active_batch))
            diagnostics.update(self._router_fit_diagnostics)
            diagnostics.update(self._legitimacy_diagnostics(decision, router_skipped=False))
            return preds.astype(np.float32), diagnostics

        tokens = self._build_regression_tokens(active_matrix, active_batch)
        self._router.eval()
        with torch.no_grad():
            token_tensor = tokens.tokens.to(self.device)
            router_out = self._router(token_tensor, full_index=active_batch.full_index)
            router_has_nonfinite = (
                (not torch.isfinite(router_out.specialist_weights).all())
                or (not torch.isfinite(router_out.defer_prob).all())
            )
            if router_has_nonfinite:
                fallback_preds = np.asarray(active_batch.predictions[:, active_batch.full_index], dtype=np.float32)
                preds = anchor_preds.copy()
                preds[active_mask] = fallback_preds
                diagnostics = {
                    "router_kind": f"{router_out.router_kind}_nonfinite_fallback",
                    "mean_defer_prob": float(torch.nanmean(router_out.defer_prob).item()),
                    "full_index": int(active_batch.full_index),
                    "router_nonfinite_fallback": True,
                    "effective_defer_rate": 0.0,
                    "regression_router_fallback_stage": "predict_router_output",
                    "regression_router_fallback_reason": "nonfinite_router_output",
                    "prediction_router_tokens_finite_flag": bool(torch.isfinite(token_tensor).all().item()),
                    "prediction_router_weights_finite_flag": bool(torch.isfinite(router_out.specialist_weights).all().item()),
                    "prediction_router_defer_finite_flag": bool(torch.isfinite(router_out.defer_prob).all().item()),
                }
                base_defaults = self._regression_fallback_defaults()
                base_defaults.update(diagnostics)
                diagnostics = base_defaults
                diagnostics.update(self._portfolio_diagnostics(active_batch))
                diagnostics.update(self._router_fit_diagnostics)
                diagnostics.update(self._legitimacy_diagnostics(decision, router_skipped=False))
                return preds.astype(np.float32), diagnostics
            integration = integrate_predictions(expert_predictions=active_batch.predictions, router_outputs=router_out)

        preds = anchor_preds.copy()
        preds[active_mask] = integration.predictions
        diagnostics = self._regression_fallback_defaults()
        diagnostics.update(integration.diagnostics)
        diagnostics.update(self._router_fit_diagnostics)
        if router_out.ot_costs is not None:
            diagnostics["mean_ot_cost"] = float(router_out.ot_costs.mean().item())
        if router_out.aux_loss is not None:
            diagnostics["alignment_aux_loss"] = float(router_out.aux_loss.detach().cpu().item())
        if router_out.extra_diagnostics:
            diagnostics.update(router_out.extra_diagnostics)
        diagnostics.update(self._task_prior_diagnostics)
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
