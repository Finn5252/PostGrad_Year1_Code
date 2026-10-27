# libraries
from __future__ import annotations

import torch
from torch import Tensor

REG_MODES = ("error", "prediction")

def laplacian_penalty(
        pred: Tensor,  # (N, n_outputs), scaled
        target: Tensor, # (N, n_outputs), scaled
        edge_index: Tensor, # (2, E) unordered pairs
        edge_weight: Tensor, # (E,) Gaussian kernel values
        mode: str = "error",
) -> Tensor:
    if mode not in REG_MODES:
        raise ValueError(f"mode must be one of {REG_MODES}, got {mode!r}")
    field = pred - target if mode == "error" else pred
    src, dst = edge_index
    # mean over the output channels
    diff2 = (field[src] - field[dst]).pow(2).mean(dim = 1)
    # weighted mean over pairs
    return (edge_weight * diff2).sum() / edge_weight.sum().clamp_min(1e-12)