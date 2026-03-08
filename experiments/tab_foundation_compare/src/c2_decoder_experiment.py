from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from .aligned_california import (
    SEED,
    build_aligned_california_split,
    denormalize,
    standardize_regression_targets,
    tabm_noisy_quantile_transform,
)
from .c2_tabm_model import DecoderConfig, GatedTabMRegressor, MeanTabMRegressor, load_tabm_module


TABM_ROOT = Path("/private/tmp/tabm_clone_inspect_20260308")


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 256
    max_epochs: int = 80
    patience: int = 16
    lr: float = 8.72489003621806e-4
    weight_decay: float = 3.777165108799435e-2
    seed: int = 0
    smoke: bool = False


@dataclass(frozen=True)
class DatasetBundle:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train_norm: np.ndarray
    y_val_norm: np.ndarray
    y_test_norm: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    target_stats: dict[str, float]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataset_bundle(smoke: bool = False, seed: int = SEED) -> DatasetBundle:
    split = build_aligned_california_split(seed=seed)
    x_train, x_val, x_test = tabm_noisy_quantile_transform(split, seed=seed)
    y_train_norm, y_val_norm, y_test_norm, target_stats = standardize_regression_targets(
        split.y_train, split.y_val, split.y_test
    )

    if smoke:
        train_n = min(4096, len(x_train))
        val_n = min(1024, len(x_val))
        test_n = min(1024, len(x_test))
        x_train = x_train[:train_n]
        x_val = x_val[:val_n]
        x_test = x_test[:test_n]
        y_train_norm = y_train_norm[:train_n]
        y_val_norm = y_val_norm[:val_n]
        y_test_norm = y_test_norm[:test_n]
        y_train = split.y_train[:train_n]
        y_val = split.y_val[:val_n]
        y_test = split.y_test[:test_n]
    else:
        y_train = split.y_train
        y_val = split.y_val
        y_test = split.y_test

    return DatasetBundle(
        x_train=x_train.astype(np.float32),
        x_val=x_val.astype(np.float32),
        x_test=x_test.astype(np.float32),
        y_train_norm=y_train_norm.astype(np.float32),
        y_val_norm=y_val_norm.astype(np.float32),
        y_test_norm=y_test_norm.astype(np.float32),
        y_train=y_train.astype(np.float32),
        y_val=y_val.astype(np.float32),
        y_test=y_test.astype(np.float32),
        target_stats=target_stats,
    )


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "score": float(-rmse),
    }


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    x: np.ndarray,
    batch_size: int,
    target_stats: dict[str, float],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    model.eval()
    preds: list[np.ndarray] = []
    gate_mass: list[np.ndarray] = []
    head_std: list[np.ndarray] = []
    for x_batch, _ in make_loader(x, np.zeros(len(x), dtype=np.float32), batch_size, shuffle=False):
        pred_norm, aux = model(x_batch.to(device))
        preds.append(pred_norm.detach().cpu().numpy())
        if "gate" in aux:
            gate_mass.append(aux["gate"].detach().cpu().numpy())
        if "head_preds" in aux:
            head_std.append(aux["head_preds"].detach().cpu().numpy().std(axis=1))
    pred = denormalize(np.concatenate(preds, axis=0), target_stats)
    diagnostics: dict[str, float] = {}
    if gate_mass:
        gate = np.concatenate(gate_mass, axis=0)
        diagnostics["gate_entropy"] = float((-gate * np.log(gate + 1e-8)).sum(axis=1).mean())
        diagnostics["gate_top1_mass"] = float(gate.max(axis=1).mean())
    if head_std:
        diagnostics["head_prediction_std"] = float(np.concatenate(head_std, axis=0).mean())
    return pred, diagnostics


def train_one(
    model_name: str,
    model: torch.nn.Module,
    dataset: DatasetBundle,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, object]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    train_loader = make_loader(dataset.x_train, dataset.y_train_norm, config.batch_size, shuffle=True)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_rmse = float("inf")
    patience_left = config.patience
    started = time.time()

    for epoch in range(config.max_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred_norm, _ = model(x_batch)
            loss = torch.nn.functional.mse_loss(pred_norm, y_batch)
            loss.backward()
            optimizer.step()

        val_pred, _ = predict(
            model,
            dataset.x_val,
            config.batch_size,
            dataset.target_stats,
            device,
        )
        val_metrics = regression_metrics(dataset.y_val, val_pred)
        if val_metrics["rmse"] < best_val_rmse - 1e-6:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    val_pred, val_diag = predict(model, dataset.x_val, config.batch_size, dataset.target_stats, device)
    test_pred, test_diag = predict(model, dataset.x_test, config.batch_size, dataset.target_stats, device)
    train_pred, train_diag = predict(
        model,
        dataset.x_train,
        config.batch_size,
        dataset.target_stats,
        device,
    )
    duration = time.time() - started
    diagnostics = {}
    diagnostics.update({f"train_{k}": v for k, v in train_diag.items()})
    diagnostics.update({f"val_{k}": v for k, v in val_diag.items()})
    diagnostics.update({f"test_{k}": v for k, v in test_diag.items()})
    return {
        "model": model_name,
        "best_epoch": best_epoch,
        "duration_seconds": round(duration, 3),
        "train": regression_metrics(dataset.y_train, train_pred),
        "val": regression_metrics(dataset.y_val, val_pred),
        "test": regression_metrics(dataset.y_test, test_pred),
        "diagnostics": diagnostics,
    }


def run_c2_decoder_experiment(output_dir: Path, smoke: bool = False) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(0)
    dataset = build_dataset_bundle(smoke=smoke, seed=SEED)
    tabm_module = load_tabm_module(TABM_ROOT)
    decoder_config = DecoderConfig(n_num_features=dataset.x_train.shape[1])
    train_config = TrainConfig(smoke=smoke, max_epochs=8 if smoke else 80, patience=4 if smoke else 16)
    device = torch.device("cpu")

    mean_result = train_one(
        "C2a_TabM_mean_heads",
        MeanTabMRegressor(tabm_module, decoder_config),
        dataset,
        train_config,
        device,
    )
    gated_result = train_one(
        "C2b_TabM_gated_heads",
        GatedTabMRegressor(tabm_module, decoder_config),
        dataset,
        train_config,
        device,
    )

    result = {
        "smoke": smoke,
        "train_config": asdict(train_config),
        "decoder_config": asdict(decoder_config),
        "results": [mean_result, gated_result],
    }
    (output_dir / ("c2_decoder_results__smoke.json" if smoke else "c2_decoder_results.json")).write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result
