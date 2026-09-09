from __future__ import annotations

import argparse
import csv
import json
import random
import time

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from data import CropBox, DataConfig, ScalerBundle, build_splits
from metrics import (HalfMeanSquaredError, MeanAccumulator, RelativeErrorAccumulator, RelativeErrorConfig)
from model import GCNSurrogate, GCNSurrogateConfig, count_parameters

# configuration

@dataclass
class TrainConfig:
    out_dir: str = "tuns/default"
    epochs: int = 1000
    batch_size: int = 1
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    lr_decay_rate: float = 0.9
    lr_decay_every_epochs: int = 200
    early_stopping_patience: int = 200
    early_stopping_min_delta: float = 0.0

    seed: int = 0
    device: str = "cpu"
    num_workers: int = 0

    relative_error: RelativeErrorConfig = field(default_factory = RelativeErrorConfig)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Epoch helpers

def _forward(model: GCNSurrogate, batch, device:str) -> tuple[torch.Tensor, torch.Tensor]:
    batch = batch.to(device)
    pred = model(batch.x, batch.edge_index, batch.scalars, batch.batch)
    return pred, batch.y

def run_epoch(model, loader, loss_fn, device, optimizer = None) -> float:
    "case averaged loss returned per one pass"
    training = optimizer is not None
    model.train(training)
    acc = MeanAccumulator()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            pred, y = forward(model, batch, device)
            loss = loss_fn(pred, y)
            if training:
                optimizer.zero_grad(set_to_none = True)
                loss.backward()
                optimizer.step()
            n_graphs = int(batch.num_graphs)
            acc.update(loss.item(), n = n_graphs)
        return acc.mean
@ torch.no_grad()
def evaluate_relative_errors(model, loader, scalars: ScalarBundle, device: str, cfg: RelativeErrorConfig) -> dict:
    "Relative L1/L2 on denormalised values, per target field"
    model.eval()
    acc = RelativeErrorAccumulator(scalars.target_columns, cfg)

    for batch in loader:
        y_true = scalars.target.inverse_transform(batch.y.to(device))
        acc.update_scale(y_true)
    acc.lock_scale()

    for batch in loader:
        pred, y = forward(model, batch, device)
        acc.update(scalars.target.inverse_transform(pred), scalars.target.inverse_transform(y),)
    return acc.result()

