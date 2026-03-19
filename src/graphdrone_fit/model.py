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
from .geo_ensemble import anchor_geo_poe_blend, learned_geo_poe_blend, learned_geo_poe_blend_torch


@dataclass(frozen=True)
class GraphDronePredictResult:
    predictions: np.ndarray
    diagnostics: dict[str, object]
    expert_ids: tuple[str, ...]


def _clf_entropy(predictions: np.ndarray) -> np.ndarray:
    """
    Convert [N, E, C] class probability array to [N, E] Shannon entropy scalars.
    Used to give the token builder a scalar signal per expert — high entropy means
    an uncertain expert, which the router should weight differently.
    """
    p = np.clip(predictions, 1e-9, 1.0)
    return -np.sum(p * np.log(p), axis=-1).astype(np.float32)


def _coerce_matrix(X: np.ndarray) -> np.ndarray:
    matrix = np.asarray(X, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {matrix.shape}")
    return matrix


def _detect_problem_type(y: np.ndarray, config: GraphDroneConfig):
    """
    Returns (problem_type, n_classes).

    Priority:
    1. config.n_classes > 1  → classification, use that n_classes
    2. Auto-detect from y:
       - float targets with non-integer values → regression
       - ≤ 50 unique integer-valued targets → classification
       - otherwise → regression
    """
    if config.n_classes > 1:
        return "classification", config.n_classes

    unique_y = np.unique(y)
    is_float_target = np.issubdtype(y.dtype, np.floating) and not np.all(np.mod(y, 1) == 0)

    if not is_float_target and len(unique_y) < 50:
        return "classification", int(unique_y.max() + 1)

    return "regression", 1


class GraphDrone:
    """
    GraphDrone v1-width + GeoPOE multiclass extension.

    Regression path:  unchanged from v1-width (100% data, GORA, static anchor router).
    Classification path: 100% data + Geometric Product-of-Experts blending — no router
                         training required, no NLL loss, valid probability output guaranteed.
    """
    def __init__(self, config: GraphDroneConfig) -> None:
        self.config = config.validate()
        self._expert_factory: Optional[PortfolioExpertFactory] = None
        self._token_builder = UniversalTokenBuilder()
        self._support_encoder = MomentSupportEncoder()
        self._router: Optional[torch.nn.Module] = None
        self._train_views: dict[str, np.ndarray] = {}
        self._y_train_reg: Optional[np.ndarray] = None  # stored for quality score computation at predict time
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
        X = np.asarray(X, dtype=np.float32)
        matrix = _coerce_matrix(X)
        self.n_features_in_ = matrix.shape[1]

        # Detect problem type -----------------------------------------------
        if problem_type is not None:
            # Legacy explicit override from benchmark script
            if problem_type in ("binary", "classification"):
                n_classes = self.config.n_classes if self.config.n_classes > 1 else int(len(np.unique(y)))
                self._problem_type = "classification"
                self._n_classes = n_classes
            else:
                self._problem_type = "regression"
                self._n_classes = 1
        else:
            self._problem_type, self._n_classes = _detect_problem_type(
                np.asarray(y), self.config
            )

        is_classification = (self._problem_type == "classification")

        # Cast y appropriately
        if is_classification:
            y = np.asarray(y, dtype=np.int64)
        else:
            y = np.asarray(y, dtype=np.float32)

        # Build default specs if not supplied --------------------------------
        if expert_specs is None:
            full_idx = tuple(range(matrix.shape[1]))
            params = {"n_estimators": 8, "device": self.device}
            rng = np.random.RandomState(42)
            sub_size = max(1, int(matrix.shape[1] * 0.7))
            sub_idx = tuple(sorted(rng.choice(matrix.shape[1], sub_size, replace=False).tolist()))

            if is_classification:
                model_kind = "foundation_classifier"
                full_spec = ExpertBuildSpec(
                    descriptor=ViewDescriptor(
                        expert_id=self.config.full_expert_id, family="FULL",
                        view_name="Foundation Full", is_anchor=True,
                        input_dim=matrix.shape[1], input_indices=full_idx,
                    ),
                    model_kind=model_kind,
                    input_adapter=IdentitySelectorAdapter(indices=full_idx),
                    model_params=params,
                )
                sub_specs = []
                for sub_seed, sub_frac in [(0, 0.7), (1, 0.7), (2, 0.8)]:
                    rng_i = np.random.RandomState(sub_seed)
                    sz_i = max(1, int(matrix.shape[1] * sub_frac))
                    idx_i = tuple(sorted(rng_i.choice(matrix.shape[1], sz_i, replace=False).tolist()))
                    sub_specs.append(ExpertBuildSpec(
                        descriptor=ViewDescriptor(
                            expert_id=f"SUB{sub_seed}", family="structural_subspace",
                            view_name=f"Foundation Sub {sub_seed}",
                            input_dim=sz_i, input_indices=idx_i,
                        ),
                        model_kind=model_kind,
                        input_adapter=IdentitySelectorAdapter(indices=idx_i),
                        model_params=params,
                    ))
                expert_specs = (full_spec, *sub_specs)
            else:
                expert_specs = (
                    ExpertBuildSpec(
                        descriptor=ViewDescriptor(
                            expert_id=self.config.full_expert_id, family="FULL",
                            view_name="Foundation Full", is_anchor=True,
                            input_dim=matrix.shape[1], input_indices=full_idx,
                        ),
                        model_kind="foundation_regressor",
                        input_adapter=IdentitySelectorAdapter(indices=full_idx),
                        model_params=params,
                    ),
                )

        print(f"  -> Fitting GraphDrone ({self._problem_type}, n_classes={self._n_classes}) on {len(X)} samples...")

        # 1. Fit experts (100% data utilization — v1-width strength) ---------
        self._portfolio = fit_portfolio_from_specs(
            X_train=matrix, y_train=y, specs=expert_specs,
            full_expert_id=self.config.full_expert_id,
        )
        self._expert_factory = PortfolioExpertFactory(self._portfolio)

        # Classification path ------------------------------------------------
        if is_classification:
            # Always populate _train_views — needed by _compute_gora_obs in predict()
            for spec in expert_specs:
                fitted_adapter = spec.input_adapter.fit(matrix)
                self._train_views[spec.descriptor.expert_id] = fitted_adapter.transform(matrix)

            use_learned = self.config.use_learned_router_for_classification and \
                          self.config.router.kind != "bootstrap_full_only"

            if not use_learned:
                print(f"  -> Classification: using static Geometric PoE blending (no router training).")
                return self

            # --- OOF Router Training -------------------------------------------
            # Experts are already fit on 100% data (self._portfolio).
            # To avoid optimism bias, train the router on genuinely held-out expert
            # predictions: fit a temporary 90% portfolio, generate OOF predictions
            # on the 10% holdout, train router there, then restore 100% inference setup.
            from sklearn.model_selection import train_test_split as _tts
            import torch.nn.functional as F

            n_all = len(matrix)
            idx_tr90, idx_va = _tts(np.arange(n_all), test_size=0.1, random_state=42)
            X_tr90, X_va = matrix[idx_tr90], matrix[idx_va]
            y_tr90, y_va = y[idx_tr90], y[idx_va]

            print(f"  -> OOF router training: fitting temporary experts on {len(X_tr90)} rows...")
            oof_portfolio = fit_portfolio_from_specs(
                X_train=X_tr90, y_train=y_tr90, specs=expert_specs,
                full_expert_id=self.config.full_expert_id,
            )
            oof_factory = PortfolioExpertFactory(oof_portfolio)

            # Build OOF train views (90% slice) — needed for GORA computation on val rows
            oof_train_views: dict[str, np.ndarray] = {}
            for spec in expert_specs:
                fitted_adapter = spec.input_adapter.fit(X_tr90)
                oof_train_views[spec.descriptor.expert_id] = fitted_adapter.transform(X_tr90)

            # Generate OOF predictions on the 10% holdout using 90%-trained experts
            va_batch = oof_factory.predict_all(X_va)
            va_enc = self._support_encoder.encode(n_rows=len(X_va), descriptors=va_batch.descriptors)

            # Temporarily swap to OOF train views for GORA (restores after)
            saved_train_views = self._train_views
            self._train_views = oof_train_views
            va_gora = self._compute_gora_obs(X_va, va_batch.descriptors)
            self._train_views = saved_train_views                        # restore 100% views

            # Token builder expects scalar predictions [N, E]; for classification
            # use per-expert Shannon entropy as the routing signal (high H → uncertain).
            va_tokens = self._token_builder.build(
                predictions=_clf_entropy(va_batch.predictions), descriptors=va_batch.descriptors,
                full_expert_id=va_batch.full_expert_id, support_encoding=va_enc,
                geometric_obs=va_gora,
            )

            token_dim = va_tokens.tokens.shape[-1]
            self._router = build_set_router(self.config.router, token_dim=token_dim).to(self.device)
            trainable_params = list(self._router.parameters())
            if not trainable_params:
                print("  -> Classification Router has no trainable parameters, skipping optimization.")
                return self

            y_va_t = torch.tensor(y_va, dtype=torch.long).to(self.device)
            # va_batch.predictions: [N, E, C] OOF class probabilities (genuinely held-out)
            log_p = torch.log(
                torch.tensor(np.clip(va_batch.predictions, 1e-9, 1.0), dtype=torch.float32).to(self.device)
            )  # [N, E, C]
            v_tokens_t = va_tokens.tokens.to(self.device)

            # Anchor NLL is a constant baseline computed on OOF expert predictions
            with torch.no_grad():
                log_p_anchor = log_p[:, va_batch.full_index, :]         # [N, C]
                anchor_nll_val = F.nll_loss(
                    F.log_softmax(log_p_anchor, dim=-1), y_va_t
                ).item()

            optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
            best_loss = float("inf")
            patience, wait = 25, 0
            print(f"  -> Optimizing Classification Router on {self.device} "
                  f"(Patience={patience}, NLL+ResidualPenalty, OOF anchor_nll={anchor_nll_val:.4f})...")

            for _ in range(500):
                self._router.train()
                optimizer.zero_grad()
                out = self._router(v_tokens_t, full_index=va_batch.full_index)
                log_q = learned_geo_poe_blend_torch(
                    log_p, out.defer_prob, out.specialist_weights, va_batch.full_index
                )  # [N, C]
                blend_nll = F.nll_loss(F.log_softmax(log_q, dim=-1), y_va_t)
                loss = blend_nll + 2.0 * F.relu(blend_nll - anchor_nll_val)
                loss.backward()
                optimizer.step()
                # Fix 2: early-stop on protected objective, not raw blend_nll alone
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
                frac_better = float((F.nll_loss(F.log_softmax(log_q_final, dim=-1), y_va_t, reduction="none")
                                     < F.nll_loss(F.log_softmax(log_p_anchor, dim=-1), y_va_t, reduction="none")
                                    ).float().mean().item())
            print(f"  -> Classification Router trained. "
                  f"blend_nll={final_blend_nll:.4f}  anchor_nll={anchor_nll_val:.4f}  "
                  f"mean_defer={mean_defer:.3f}  frac_rows_blend_wins={frac_better:.2f}")

            self._clf_uses_learned_router = True
            return self

        # Regression path: GORA + quality scores + learned router -----------
        for spec in expert_specs:
            fitted_adapter = spec.input_adapter.fit(matrix)
            self._train_views[spec.descriptor.expert_id] = fitted_adapter.transform(matrix)

        # Store y_train for quality score computation at predict time
        self._y_train_reg = y

        from sklearn.model_selection import train_test_split
        _, X_va, _, y_va = train_test_split(X, y, test_size=0.1, random_state=42)

        va_batch = self._expert_factory.predict_all(X_va)
        va_enc = self._support_encoder.encode(n_rows=len(X_va), descriptors=va_batch.descriptors)
        va_gora = self._compute_gora_obs(X_va, va_batch.descriptors)
        va_quality = self._compute_quality_obs(X_va, va_batch.descriptors)

        va_tokens = self._token_builder.build(
            predictions=va_batch.predictions, descriptors=va_batch.descriptors,
            full_expert_id=va_batch.full_expert_id, support_encoding=va_enc,
            quality_scores=va_quality,
            geometric_obs=va_gora,
        )

        token_dim = va_tokens.tokens.shape[-1]
        self._router = build_set_router(self.config.router, token_dim=token_dim).to(self.device)
        trainable_params = list(self._router.parameters())
        if not trainable_params:
            print("  -> Router has no trainable parameters (BootstrapFullRouter), skipping optimization.")
            return self

        optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
        best_loss = float("inf")
        patience, wait = 25, 0
        y_va_t = torch.tensor(y_va).float().to(self.device)
        v_preds_t = torch.tensor(va_batch.predictions).float().to(self.device)
        v_tokens_t = va_tokens.tokens.to(self.device)

        import torch.nn.functional as F
        with torch.no_grad():
            anchor_mse_val = F.mse_loss(
                v_preds_t[:, va_batch.full_index], y_va_t
            ).item()
        print(f"  -> Optimizing Router on {self.device} "
              f"(Patience={patience}, MSE+ResidualPenalty, anchor_mse={anchor_mse_val:.6f})...")

        for _ in range(500):
            self._router.train()
            optimizer.zero_grad()
            out = self._router(v_tokens_t, full_index=va_batch.full_index)
            integ = (
                (1 - out.defer_prob) * v_preds_t[:, va_batch.full_index : va_batch.full_index + 1]
                + out.defer_prob * (out.specialist_weights * v_preds_t).sum(dim=1, keepdim=True)
            )
            mse = F.mse_loss(integ.squeeze(), y_va_t)
            loss = mse + 2.0 * F.relu(mse - anchor_mse_val)
            loss.backward()
            optimizer.step()
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
            X_v = X[:, list(d.input_indices)] if d.input_indices else X
            knn = NearestNeighbors(n_neighbors=d.preferred_k).fit(X_tr_v)
            dists, indices = knn.kneighbors(X_v)
            kappa = calculate_kappa(X_tr_v, indices).reshape(-1, 1)
            lid = calculate_lid(dists).reshape(-1, 1)
            all_obs.append(np.concatenate([kappa, lid], axis=1))

        return torch.tensor(np.stack(all_obs, axis=1), dtype=torch.float32)

    def _compute_quality_obs(self, X: np.ndarray, descriptors: tuple[ViewDescriptor, ...]) -> torch.Tensor:
        """
        Per-expert local label uncertainty for regression.

        For each test sample × expert view, finds the k nearest training neighbors
        in that expert's feature subspace and computes std of their y_train labels.
        High std = noisy/uncertain region for this expert = lower quality.

        Returns [N, E, 1] tensor: -log1p(local_label_std) per expert per sample.
        More negative = less reliable expert here. Zero = perfectly smooth region.

        Shares training views with _compute_gora_obs but uses y labels, not geometry.
        Only valid for regression (requires self._y_train_reg).
        """
        from sklearn.neighbors import NearestNeighbors

        y_ref = self._y_train_reg  # [N_train] full training labels
        all_scores = []
        for d in descriptors:
            X_tr_v = self._train_views[d.expert_id]
            X_v = X[:, list(d.input_indices)] if d.input_indices else X
            k = min(d.preferred_k, len(X_tr_v) - 1)
            knn = NearestNeighbors(n_neighbors=k).fit(X_tr_v)
            _, indices = knn.kneighbors(X_v)
            # std of training labels in this expert's local neighborhood
            local_std = np.std(y_ref[indices], axis=1)  # [N]
            # Log-scale and negate: higher std → more negative → lower quality
            # log1p prevents -inf at std=0 (perfectly smooth region → score=0)
            quality = -np.log1p(local_std).reshape(-1, 1)  # [N, 1]
            all_scores.append(quality)
        scores = np.stack(all_scores, axis=1)  # [N, E, 1]
        return torch.tensor(scores, dtype=torch.float32)

    def predict(self, X: np.ndarray, return_diagnostics: bool = False) -> Union[np.ndarray, GraphDronePredictResult]:
        X = np.asarray(X, dtype=np.float32)
        matrix = _coerce_matrix(X)
        batch = self._expert_factory.predict_all(matrix)

        # --- Classification path --------------------------------------------
        if self._problem_type == "classification":
            if self._clf_uses_learned_router and self._router is not None:
                # Learned anchor-protective router path
                support_enc = self._support_encoder.encode(
                    n_rows=matrix.shape[0], descriptors=batch.descriptors
                )
                gora_obs = self._compute_gora_obs(matrix, batch.descriptors)
                tokens = self._token_builder.build(
                    predictions=_clf_entropy(batch.predictions), descriptors=batch.descriptors,
                    full_expert_id=batch.full_expert_id, support_encoding=support_enc,
                    geometric_obs=gora_obs,
                )
                self._router.eval()
                with torch.no_grad():
                    token_tensor = tokens.tokens.to(self.device)
                    router_out = self._router(token_tensor, full_index=batch.full_index)

                preds = learned_geo_poe_blend(
                    batch.predictions,
                    router_out.defer_prob,
                    router_out.specialist_weights,
                    batch.full_index,
                )
                diagnostics = {
                    "router_kind": router_out.router_kind,
                    "mean_defer_prob": float(router_out.defer_prob.mean().item()),
                }
            else:
                # Static anchor-boosted GeoPOE (anchor_weight=3.0 default)
                preds = anchor_geo_poe_blend(
                    batch.predictions,
                    anchor_idx=batch.full_index,
                )
                diagnostics = {
                    "router_kind": "geo_poe",
                    "mean_defer_prob": float("nan"),
                }

            if return_diagnostics:
                return GraphDronePredictResult(
                    predictions=preds,
                    expert_ids=batch.expert_ids,
                    diagnostics=diagnostics,
                )
            return preds

        # --- Regression path: GORA + quality scores + learned router --------
        support_enc = self._support_encoder.encode(
            n_rows=matrix.shape[0], descriptors=batch.descriptors
        )
        gora_obs = self._compute_gora_obs(matrix, batch.descriptors)
        quality_obs = self._compute_quality_obs(matrix, batch.descriptors)

        tokens = self._token_builder.build(
            predictions=batch.predictions,
            descriptors=batch.descriptors,
            full_expert_id=batch.full_expert_id,
            support_encoding=support_enc,
            quality_scores=quality_obs,
            geometric_obs=gora_obs,
        )

        self._router.eval()
        with torch.no_grad():
            token_tensor = tokens.tokens.to(self.device)
            router_out = self._router(token_tensor, full_index=batch.full_index)
            integration = integrate_predictions(
                expert_predictions=batch.predictions, router_outputs=router_out
            )

        if return_diagnostics:
            return GraphDronePredictResult(
                predictions=integration.predictions,
                expert_ids=batch.expert_ids,
                diagnostics=integration.diagnostics,
            )
        return integration.predictions
