from __future__ import annotations

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

#settings:

H5_PATH = r"data/hydrofoil.h5"      
OUT_DIR = "runs/run1"               
 
CROP = (-0.5, 2.0, 0.6, 1.4)        
KNN_K = 4
CACHE_DIR = "cache"                 
 
EPOCHS = 1000
BATCH_SIZE = 1                      
LR = 1e-3
PATIENCE = 200                      
SEED = 0
 
HIDDEN = 256
SHARED_BLOCKS = 4
GCN_BLOCKS = 3
 
ZERO_HANDLING = "mask"              
ZERO_THRESHOLD = 1e-3
 

SMOKE = False                       


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

# training

def train(
        data_cfg: DataConfig,
        model_cfg: GCNSurrogateConfig,
        train_cfg: TrainConfig
) -> dict: 
    out = Path(train_cfg.out_dir)
    out.mkdir(parents = True, exist_ok = True)
    set_seed(train_cfg.seed)
    device = train_cfg.device

    train_ds, val_ds, test_ds, scalars, store = build_splits(data_cfg)
    scalars.save(out / "scalars.json")

    train_loader = DataLoader(train_ds, batch_size = train_cfg.batch_size, shuffle = True)
    val_loader = DataLoader(val_ds, batch_size = train_cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size = train_cfg.batch_size)

    model = GCNSurrogate(model_cfg).to(device)
    print(f"[train] {count_parameters(model):,} trainable parameters on {device}")

    loss_fn = HalfMeanSquaredError()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr = train_cfg.lr,
        betas = train_cfg.betas,
        weight_decay = train_cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size = train_cfg.lr_decay_every_epochs, gamma = train_cfg.lr_decay_rate
    )

    (out / "config.json").write_text(
        json.dumps(
            {
                "data": _jsonable(asdict(data_cfg)),
                "model": _jsonable(asdict(model_cfg)),
                "train": _jsonable(asdict(train_cfg)),
                "n_parameters": count_parameters(model),
            }  
        )
    )

    history_path = out / "history.csv"
    ckpt_path = out / "best.pt"
    best_val = float("inf")
    best_epoch = -1
    epochs_since_improvement = 0
    t0 = time.time()

    with history_path.open("w", newline = "") as fh:
        writer = csv.DictWriter(
            fh, fieldnames = ["epoch", "lr", "train_loss", "val_loss", "test_loss"]
        )
        writer.writeheader()

        for epoch in range(1, train_cfg.epochs + 1):
            lr_now = optimizer.param_groups[0]["lr"]
            train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)
            val_loss = run_epoch(model, val_loader, loss_fn, device)
            test_loss = (
                run_epoch(model, test_loader, loss_fn, device)
                if train_cfg.log_test_each_epoch
                else float("nan")
            )
            scheduler.step()

            writer.writerow(
                {
                    "epoch": epoch,
                    "lr": lr_now,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "test_loss": test_loss,
                }
            )
            fh.flush()
            
            writer.writerow(
                {
                    "epoch": epoch,
                    "lr": lr_now,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "test_loss": test_loss,
                }
            )
            fh.flush()

