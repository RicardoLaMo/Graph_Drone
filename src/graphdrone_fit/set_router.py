from __future__ import annotations

from dataclasses import dataclass
import torch

from .config import SetRouterConfig


@dataclass(frozen=True)
class RouterOutputs:
    specialist_weights: torch.Tensor
    defer_prob: torch.Tensor
    full_index: int
    router_kind: str


class BootstrapFullRouter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_fit_summary: dict[str, object] = {
            "router_kind": "bootstrap_full_only",
            "fit_status": "skipped",
        }

    def fit_router(
        self,
        tokens: torch.Tensor,
        expert_predictions: torch.Tensor,
        y_true: torch.Tensor,
        *,
        full_index: int,
        field_slices: dict[str, tuple[int, int]] | None = None,
    ) -> dict[str, object]:
        del tokens, expert_predictions, y_true, full_index, field_slices
        self.last_fit_summary = {
            "router_kind": "bootstrap_full_only",
            "fit_status": "skipped",
        }
        return dict(self.last_fit_summary)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        full_index: int,
        field_slices: dict[str, tuple[int, int]] | None = None,
    ) -> RouterOutputs:
        del field_slices
        if tokens.ndim != 3:
            raise ValueError(f"Expected token tensor [N, E, D], got {tuple(tokens.shape)}")
        n_rows, n_experts, _ = tokens.shape
        specialist_weights = torch.zeros((n_rows, n_experts), dtype=tokens.dtype, device=tokens.device)
        defer_prob = torch.zeros((n_rows, 1), dtype=tokens.dtype, device=tokens.device)
        return RouterOutputs(
            specialist_weights=specialist_weights,
            defer_prob=defer_prob,
            full_index=full_index,
            router_kind="bootstrap_full_only",
        )


class ContextualSparseRouter(torch.nn.Module):
    def __init__(self, config: SetRouterConfig) -> None:
        super().__init__()
        self.config = config
        self.token_encoder: torch.nn.Sequential | None = None
        self.field_projections: torch.nn.ModuleDict | None = None
        self.field_fusion: torch.nn.Sequential | None = None
        self._field_signature: tuple[tuple[str, int, int], ...] | None = None
        self.specialist_head: torch.nn.Sequential | None = None
        self.defer_head: torch.nn.Sequential | None = None
        self.last_fit_summary: dict[str, object] = {
            "router_kind": "contextual_sparse_mlp",
            "fit_status": "unfitted",
        }

    def _ensure_modules(
        self,
        token_dim: int,
        *,
        field_slices: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        if self.token_encoder is not None or self.field_fusion is not None:
            return
        hidden_dim = self.config.hidden_dim
        signature = _normalize_field_signature(field_slices)
        if self.config.field_aware_tokens and signature:
            self._field_signature = signature
            self.field_projections = torch.nn.ModuleDict(
                {
                    name: torch.nn.Sequential(
                        torch.nn.Linear(stop - start, hidden_dim),
                        torch.nn.ReLU(),
                    )
                    for name, start, stop in signature
                    if stop > start
                }
            )
            self.field_fusion = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim * len(self.field_projections), hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
            )
        else:
            self.token_encoder = torch.nn.Sequential(
                torch.nn.Linear(token_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
            )
        self.specialist_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.defer_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 3, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        with torch.no_grad():
            final = self.defer_head[-1]
            if isinstance(final, torch.nn.Linear):
                final.bias.fill_(float(self.config.defer_bias))

    def fit_router(
        self,
        tokens: torch.Tensor,
        expert_predictions: torch.Tensor,
        y_true: torch.Tensor,
        *,
        full_index: int,
        field_slices: dict[str, tuple[int, int]] | None = None,
    ) -> dict[str, object]:
        tokens = torch.as_tensor(tokens, dtype=torch.float32)
        expert_predictions = torch.as_tensor(expert_predictions, dtype=torch.float32)
        if tokens.ndim != 3:
            raise ValueError(f"Expected token tensor [N, E, D], got {tuple(tokens.shape)}")
        if expert_predictions.ndim == 2:
            expected = tokens.shape[:2]
            if expert_predictions.shape != expected:
                raise ValueError(
                    f"Expected expert_predictions shape {tuple(expected)}, got {tuple(expert_predictions.shape)}"
                )
            y_true = torch.as_tensor(y_true, dtype=torch.float32).reshape(-1)
        elif expert_predictions.ndim == 3:
            expected = (tokens.shape[0], tokens.shape[1])
            if expert_predictions.shape[:2] != expected:
                raise ValueError(
                    f"Expected expert_predictions leading shape {tuple(expected)}, got {tuple(expert_predictions.shape[:2])}"
                )
            y_true = torch.as_tensor(y_true, dtype=torch.long).reshape(-1)
        else:
            raise ValueError(
                f"Expected expert_predictions rank 2 or 3, got {tuple(expert_predictions.shape)}"
            )
        if y_true.shape[0] != tokens.shape[0]:
            raise ValueError(f"Expected y_true with {tokens.shape[0]} rows, got {y_true.shape[0]}")

        self._ensure_modules(tokens.shape[-1], field_slices=field_slices)
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )

        train_index, val_index = _split_router_indices(
            n_rows=tokens.shape[0],
            validation_fraction=float(self.config.validation_fraction),
            seed=int(self.config.random_seed),
        )
        best_state = {key: value.detach().clone() for key, value in self.state_dict().items()}
        best_val_loss = float("inf")
        best_epoch = -1
        patience_left = int(self.config.patience)
        history: list[dict[str, float]] = []

        train_tokens = tokens[train_index]
        train_preds = expert_predictions[train_index]
        train_targets = y_true[train_index]
        val_tokens = tokens[val_index]
        val_preds = expert_predictions[val_index]
        val_targets = y_true[val_index]

        for epoch in range(int(self.config.max_epochs)):
            self.train()
            optimizer.zero_grad(set_to_none=True)
            outputs = self._forward_router(train_tokens, full_index=full_index, field_slices=field_slices)
            train_loss = _router_task_loss(
                router_outputs=outputs,
                expert_predictions=train_preds,
                y_true=train_targets,
            )
            train_loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                val_outputs = self._forward_router(val_tokens, full_index=full_index, field_slices=field_slices)
                val_loss = _router_task_loss(
                    router_outputs=val_outputs,
                    expert_predictions=val_preds,
                    y_true=val_targets,
                )
            train_value = float(train_loss.detach().cpu().item())
            val_value = float(val_loss.detach().cpu().item())
            history.append({"epoch": float(epoch), "train_loss": train_value, "val_loss": val_value})

            if val_value + 1e-8 < best_val_loss:
                best_val_loss = val_value
                best_epoch = epoch
                best_state = {key: value.detach().clone() for key, value in self.state_dict().items()}
                patience_left = int(self.config.patience)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        self.load_state_dict(best_state)
        self.eval()
        with torch.no_grad():
            outputs = self._forward_router(tokens, full_index=full_index, field_slices=field_slices)
            specialist_mass = outputs.specialist_weights.sum(dim=1)
            top_specialist = outputs.specialist_weights.argmax(dim=1)
            active_specialist = specialist_mass > 0
            top_specialist_fraction = float(active_specialist.float().mean().item())
            unique_specialists = sorted(
                {
                    int(index)
                    for index in top_specialist[active_specialist].detach().cpu().tolist()
                    if int(index) != full_index
                }
            )

        self.last_fit_summary = {
            "router_kind": "contextual_sparse_mlp",
            "fit_status": "fitted",
            "train_rows": int(train_index.numel()),
            "validation_rows": int(val_index.numel()),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "history_tail": history[-5:],
            "mean_defer_prob": float(outputs.defer_prob.mean().item()),
            "mean_specialist_mass": float(outputs.specialist_weights.sum(dim=1).mean().item()),
            "top_specialist_fraction": top_specialist_fraction,
            "active_specialist_indices": unique_specialists,
            "sparse_top_k": int(self.config.sparse_top_k),
            "token_encoder_kind": "field_aware" if self.field_fusion is not None else "flat_mlp",
        }
        return dict(self.last_fit_summary)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        full_index: int,
        field_slices: dict[str, tuple[int, int]] | None = None,
    ) -> RouterOutputs:
        tokens = torch.as_tensor(tokens, dtype=torch.float32)
        if tokens.ndim != 3:
            raise ValueError(f"Expected token tensor [N, E, D], got {tuple(tokens.shape)}")
        self._ensure_modules(tokens.shape[-1], field_slices=field_slices)
        return self._forward_router(tokens, full_index=full_index, field_slices=field_slices)

    def _forward_router(
        self,
        tokens: torch.Tensor,
        *,
        full_index: int,
        field_slices: dict[str, tuple[int, int]] | None = None,
    ) -> RouterOutputs:
        if self.specialist_head is None or self.defer_head is None:
            raise RuntimeError("ContextualSparseRouter modules are not initialized")
        if not 0 <= full_index < tokens.shape[1]:
            raise ValueError(f"full_index={full_index} is out of bounds for {tokens.shape[1]} experts")

        encoded = self._encode_tokens(tokens, field_slices=field_slices)
        full_token = encoded[:, full_index : full_index + 1, :]
        global_token = encoded.mean(dim=1, keepdim=True)
        token_delta = encoded - full_token
        global_delta = encoded - global_token
        pair_features = torch.cat([encoded, token_delta, global_delta, token_delta.abs()], dim=-1)
        specialist_logits = self.specialist_head(pair_features).squeeze(-1)
        specialist_logits[:, full_index] = torch.finfo(specialist_logits.dtype).min
        sparse_logits = _apply_top_k_mask(
            specialist_logits,
            full_index=full_index,
            top_k=int(self.config.sparse_top_k),
        )
        specialist_weights = torch.softmax(sparse_logits, dim=1)
        non_full_mask = torch.ones_like(specialist_weights)
        non_full_mask[:, full_index] = 0.0
        specialist_weights = specialist_weights * non_full_mask
        specialist_mass = specialist_weights.sum(dim=1, keepdim=True)
        specialist_weights = torch.where(
            specialist_mass > 0,
            specialist_weights / specialist_mass.clamp_min(1e-8),
            torch.zeros_like(specialist_weights),
        )

        pooled = torch.cat(
            [
                full_token.squeeze(1),
                global_token.squeeze(1),
                token_delta.abs().mean(dim=1),
            ],
            dim=1,
        )
        defer_prob = torch.sigmoid(self.defer_head(pooled))
        return RouterOutputs(
            specialist_weights=specialist_weights,
            defer_prob=defer_prob,
            full_index=full_index,
            router_kind="contextual_sparse_mlp",
        )

    def _encode_tokens(
        self,
        tokens: torch.Tensor,
        *,
        field_slices: dict[str, tuple[int, int]] | None = None,
    ) -> torch.Tensor:
        if self.field_fusion is not None and self.field_projections is not None:
            signature = _normalize_field_signature(field_slices)
            if signature != self._field_signature:
                raise ValueError("Field-aware router received mismatched field_slices")
            projected_parts: list[torch.Tensor] = []
            for name, start, stop in self._field_signature or ():
                projected_parts.append(self.field_projections[name](tokens[:, :, start:stop]))
            return self.field_fusion(torch.cat(projected_parts, dim=-1))
        if self.token_encoder is None:
            raise RuntimeError("ContextualSparseRouter token encoder is not initialized")
        return self.token_encoder(tokens)


def _apply_top_k_mask(logits: torch.Tensor, *, full_index: int, top_k: int) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"Expected logits [N, E], got {tuple(logits.shape)}")
    if logits.shape[1] <= 1:
        return logits
    non_full_count = logits.shape[1] - 1
    k = max(1, min(int(top_k), non_full_count))
    masked = logits.clone()
    masked[:, full_index] = torch.finfo(masked.dtype).min
    values, indices = torch.topk(masked, k=k, dim=1)
    keep_mask = torch.zeros_like(masked, dtype=torch.bool)
    keep_mask.scatter_(1, indices, True)
    keep_mask[:, full_index] = False
    floor = torch.full_like(masked, torch.finfo(masked.dtype).min)
    return torch.where(keep_mask, masked, floor)


def _normalize_field_signature(
    field_slices: dict[str, tuple[int, int]] | None,
) -> tuple[tuple[str, int, int], ...]:
    if not field_slices:
        return ()
    items = [
        (name, int(start), int(stop))
        for name, (start, stop) in field_slices.items()
        if int(stop) > int(start)
    ]
    items.sort(key=lambda item: item[1])
    return tuple(items)


def _router_task_loss(
    *,
    router_outputs: RouterOutputs,
    expert_predictions: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    if expert_predictions.ndim == 2:
        full_pred = expert_predictions[:, router_outputs.full_index : router_outputs.full_index + 1]
        specialist_pred = (router_outputs.specialist_weights * expert_predictions).sum(dim=1, keepdim=True)
        blended = (1.0 - router_outputs.defer_prob) * full_pred + router_outputs.defer_prob * specialist_pred
        return torch.mean((blended.squeeze(1) - y_true) ** 2)
    if expert_predictions.ndim != 3:
        raise ValueError(
            f"Expected expert_predictions rank 2 or 3, got {tuple(expert_predictions.shape)}"
        )
    full_pred = expert_predictions[:, router_outputs.full_index, :]
    specialist_pred = torch.einsum("ne,nec->nc", router_outputs.specialist_weights, expert_predictions)
    blended = (1.0 - router_outputs.defer_prob) * full_pred + router_outputs.defer_prob * specialist_pred
    blended = blended.clamp_min(1e-8)
    blended = blended / blended.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return torch.nn.functional.nll_loss(blended.log(), y_true)


def _split_router_indices(*, n_rows: int, validation_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if n_rows < 4 or validation_fraction <= 0:
        indices = torch.arange(n_rows, dtype=torch.long)
        return indices, indices
    val_count = int(round(n_rows * validation_fraction))
    val_count = max(1, min(val_count, n_rows - 1))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(n_rows, generator=generator)
    val_index = perm[:val_count]
    train_index = perm[val_count:]
    return train_index, val_index


def build_set_router(config: SetRouterConfig) -> torch.nn.Module:
    config.validate()
    if config.kind == "bootstrap_full_only":
        return BootstrapFullRouter()
    if config.kind == "contextual_sparse_mlp":
        return ContextualSparseRouter(config)
    raise ValueError(f"Unsupported router kind={config.kind!r}")
