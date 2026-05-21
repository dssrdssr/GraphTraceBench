from __future__ import annotations

from typing import Dict

import torch


def regression_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    tolerance: float = 0.5,
) -> Dict[str, float]:
    preds = preds.detach().view(-1).cpu()
    targets = targets.detach().view(-1).cpu()
    errors = preds - targets
    mae = errors.abs().mean().item()
    mse = (errors ** 2).mean().item()
    rmse = mse ** 0.5
    acc_tol = (errors.abs() <= tolerance).float().mean().item()
    acc_round = (preds.round() == targets.round()).float().mean().item()
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "acc_tol": acc_tol,
        "acc_round": acc_round,
    }
