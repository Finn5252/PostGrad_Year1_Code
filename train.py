# libraries
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
import torch.nn as nn
from torch_geometric.loader import DataLoader

from data import CropBox, DataConfig, ScalerBundle, build_splits
from metrics import (MeanAccumulator, RelativeErrorAccumulator, RelativeErrorConfig)
from model import GCNSurrogate, GCNSurrogateConfig, count_parameters
from regularisation import REG_MODES, laplacian_penalty

#settings

H5_PATH = r"C:\Users\26664984\Documents\Masters\hdf5_training_data\iteration2\hydrofoil_rev2.h5"
OUT_DIR = r"C:\Users\26664984\Documents\Masters\Model_training\regularisation"


CROP = (-0.2, 1.50, 0.80, 1.4) # xmin, xmax, ymin, ymax
KNN_K = 4 # k nearest neighbours for gcn
CACHE_DIR = "cache" # KNN graphs

EPOCHS = 1000
BATCH_SIZE = 1 # number of batches during training
LR = 1e-3
PATIENCE = 200  # how many epochs validation loss must see no improvement for before early stopping
SEED = 0 # for replicability

HIDDEN = 256
SHARED_BLOCKS = 4
GCN_BLOCKS = 3 

# smoothness regularization
REG_K = 12 # neighbours in the gaussian graph for the smoothing loss
REG_LAMBDA = 0.25 # 0 = baseline run
REG_MODE = "error" # "error" penalises roughness of prediction - target

ZERO_HANDLING = "mask"   
ZERO_THRESHOLD = 1e-3

LOG_TEST_EACH_EPOCH = False # if true, test curve alongside train and validation loss

SMOKE = False  # short test runs


# configuration

@dataclass
class TrainConfig:
    out_dir: str = OUT_DIR
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    lr: float = LR
    betas: tuple[float, float] = (0.9, 0.999)  # Adam momentum and scaling decay
    weight_decay: float = 0.0           
    lr_decay_rate: float = 0.9      
    lr_decay_every_epochs: int = 200  
    early_stopping_patience: int = PATIENCE
    early_stopping_min_delta: float = 0.0

    reg_lambda: float = REG_LAMBDA
    reg_mode: str = REG_MODE

    seed: int = SEED
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    log_test_each_epoch: bool = LOG_TEST_EACH_EPOCH

    relative_error: RelativeErrorConfig = field(
        default_factory = lambda: RelativeErrorConfig(
            mode = ZERO_HANDLING, threshold = ZERO_THRESHOLD
        )
    )

    def __post_init__(self) -> None:
        if self.reg_lambda < 0:
            raise ValueError("reg_lambda must be >= 0")
        if self.reg_mode not in REG_MODES:
            raise ValueError(f"reg_mode must be one of {REG_MODES}, got {self.reg_mode!r}")

def set_seed(seed: int) -> None:
    "Four separate generators, because each library keeps its own"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Epoch helpers

def _forward(model: GCNSurrogate, batch, device:str):
    # data onto gpu
    # whole batch returned, the regularizer needs reg_edge_index and its weights
    batch = batch.to(device)
    pred = model(batch.x, batch.edge_index, batch.scalars, batch.batch)
    return pred, batch

def _losses(pred, batch, reg_lambda: float, reg_mode: str):
    "Total loss, data term and smoothness term"
    data_loss = 0.5 * nn.functional.mse_loss(pred, batch.y)
    if "reg_edge_index" not in batch:
        return data_loss, data_loss, None
    args = (pred, batch.y, batch.reg_edge_index, batch.reg_edge_weight, reg_mode)
    if reg_lambda > 0:
        reg = laplacian_penalty(*args)
        return data_loss + reg_lambda * reg, data_loss, reg
    # lambda = 0: the term is still measured
    with torch.no_grad():
        reg = laplacian_penalty(*args)
    return data_loss, data_loss, reg

def run_epoch(model, loader, device, reg_lambda: float, reg_mode: str, optimizer = None) -> dict:
    "One complete pass over a dataset, returning the case-averaged total, data and smoothness terms."
    training = optimizer is not None
    model.train(training)
    acc = {name: MeanAccumulator() for name in ("loss", "data", "reg")}
    has_reg = False
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            pred, batch = _forward(model, batch, device)
            loss, data_loss, reg = _losses(pred, batch, reg_lambda, reg_mode)
            if training:
                # clear the previous gradients, adjust weights
                optimizer.zero_grad(set_to_none = True)
                loss.backward()
                optimizer.step()
            n_graphs = int(batch.num_graphs)
            # extracts the number only for storage efficiency
            acc["loss"].update(loss.item(), n = n_graphs)
            acc["data"].update(data_loss.item(), n = n_graphs)
            if reg is not None:
                acc["reg"].update(reg.item(), n = n_graphs)
                has_reg = True
    return {
        "loss": acc["loss"].mean,
        "data": acc["data"].mean,
        "reg": acc["reg"].mean if has_reg else float("nan"),
    }

@torch.no_grad()
def evaluate_relative_errors(model, loader, scalers: ScalerBundle, device: str, cfg: RelativeErrorConfig) -> dict:
    "Relative L1/L2 on denormalised values, per target field."
    model.eval()
    acc = RelativeErrorAccumulator(scalers.target_columns, cfg)

    # pass 1 - field scale
    for batch in loader:
        y_true = scalers.target.inverse_transform(batch.y.to(device))
        acc.update_scale(y_true)
    acc.lock_scale()

    # pass 2 - errors, in Pa and m/s 
    for batch in loader:
        pred, batch = _forward(model, batch, device)
        acc.update(scalers.target.inverse_transform(pred), scalers.target.inverse_transform(batch.y),)
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
    lam, mode = train_cfg.reg_lambda, train_cfg.reg_mode

    if lam > 0 and data_cfg.reg_k is None:
        raise ValueError("reg_lambda > 0 needs a regularization graph: set reg_k in DataConfig")

    train_ds, val_ds, test_ds, scalers, store = build_splits(data_cfg)
    scalers.save(out / "scalers.json")

    # shuffle on training only 
    train_loader = DataLoader(train_ds, batch_size = train_cfg.batch_size, shuffle = True)
    val_loader = DataLoader(val_ds, batch_size = train_cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size = train_cfg.batch_size)

    model = GCNSurrogate(model_cfg).to(device)
    print(f"[train] {count_parameters(model):,} trainable parameters on {device}")
    print(f"[train] smoothness regularization: lambda={lam}, mode={mode!r}, reg_k={data_cfg.reg_k}")

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr = train_cfg.lr,
        betas = train_cfg.betas,
        weight_decay = train_cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size = train_cfg.lr_decay_every_epochs, gamma = train_cfg.lr_decay_rate
    )

    # Every setting written out before training starts
    (out / "config.json").write_text(
        json.dumps(
            {
                "data": _jsonable(asdict(data_cfg)),
                "model": _jsonable(asdict(model_cfg)),
                "train": _jsonable(asdict(train_cfg)),
                "n_parameters": count_parameters(model),
            },
            indent = 2,
        )
    )

    history_path = out / "history.csv"
    ckpt_path = out / "best.pt"
    best_val = float("inf")
    best_epoch = -1
    epochs_since_improvement = 0
    t0 = time.time()

    # loss = data + lambda * reg
    fieldnames = ["epoch", "lr"] + [
        f"{split}_{part}" for split in ("train", "val", "test") for part in ("loss", "data", "reg")
    ]
    nan_row = {"loss": float("nan"), "data": float("nan"), "reg": float("nan")}

    with history_path.open("w", newline = "") as fh:
        writer = csv.DictWriter(fh, fieldnames = fieldnames)
        writer.writeheader()

        for epoch in range(1, train_cfg.epochs + 1):
            lr_now = optimizer.param_groups[0]["lr"]
            tr = run_epoch(model, train_loader, device, lam, mode, optimizer)
            va = run_epoch(model, val_loader, device, lam, mode)
            te = (
                run_epoch(model, test_loader, device, lam, mode)
                if train_cfg.log_test_each_epoch
                else nan_row
            )
            scheduler.step()

            row = {"epoch": epoch, "lr": lr_now}
            for split, res in (("train", tr), ("val", va), ("test", te)):
                for part, value in res.items():
                    row[f"{split}_{part}"] = value
            writer.writerow(row)
            # flushed every epoch, therefore a crash still leaves a usable history
            fh.flush()

            # Selection and checkpointing based on validation loss only
            val_loss = va["loss"]
            if val_loss < best_val - train_cfg.early_stopping_min_delta:
                best_val, best_epoch = val_loss, epoch
                epochs_since_improvement = 0
                # config travels with the weights
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "model_config": asdict(model_cfg),
                        "val_loss": val_loss,
                        "reg_lambda": lam,
                        "reg_mode": mode,
                    },
                    ckpt_path,
                )
            else:
                epochs_since_improvement += 1

            if epoch == 1 or epoch % 10 == 0 or epoch == train_cfg.epochs:
                print(
                    f"[{epoch:>4}/{train_cfg.epochs}] lr={lr_now:.2e} "
                    f"train={tr['loss']:.6e} val={va['loss']:.6e} "
                    f"(val data={va['data']:.3e} reg={va['reg']:.3e}) "
                    f"best_val={best_val:.6e}@{best_epoch}"
                )

            if epochs_since_improvement >= train_cfg.early_stopping_patience:
                print(
                    f"[train] early stopping at epoch {epoch}: no validation improvement "
                    f"for {train_cfg.early_stopping_patience} epochs"
                )
                break

    # single, final test evaluation on the best checkpoint
    # Reload best checkpoint.
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location = device)["model_state"])
    model.eval()

    results = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "wall_time_s": time.time() - t0,
        "n_parameters": count_parameters(model),
        "reg": {"lambda": lam, "mode": mode, "k": data_cfg.reg_k},
        "n_cases": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "loss": {
            split: run_epoch(model, loader, device, lam, mode)
            for split, loader in (
                ("train", train_loader), ("val", val_loader), ("test", test_loader)
            )
        },
        "relative_errors": {
            split: evaluate_relative_errors(
                model, loader, scalers, device, train_cfg.relative_error
            )
            for split, loader in (
                ("train", train_loader), ("val", val_loader), ("test", test_loader)
            )
        },
    }
    (out / "results.json").write_text(json.dumps(results, indent = 2))

    print(f"\n[train] best epoch {best_epoch} (val loss {best_val:.6e})")
    t = results["loss"]["test"]
    print(
        f"[train] final test (evaluated once): loss={t['loss']:.6e} "
        f"data={t['data']:.6e} reg={t['reg']:.6e}"
    )
    for split in ("train", "val", "test"):
        print(f"[train] relative errors ({split}):")
        for name, r in results["relative_errors"][split].items():
            print(
                f"    {name:<20} L1={r['L1']:.4f}  L2={r['L2']:.4f}  "
                f"excluded={r['excluded_fraction']:.3%}  ({r['handling']})"
            )
    print(f"[train] wrote {out}/results.json, history.csv, best.pt, scalers.json")
    store.close()
    return results

def _jsonable(obj):
    "Tuples become lists, recursively (JSON has no tuple type)"
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj

def main() -> dict:
    data_cfg = DataConfig(
        h5_path = H5_PATH,
        # CROP unpacks the four-tuple into four separate arguments
        crop = CropBox(*CROP) if CROP else None,
        knn_k = KNN_K,
        reg_k = REG_K,
        cache_dir = CACHE_DIR,
        split_seed = SEED,
    )
    model_cfg = GCNSurrogateConfig(
        hidden = HIDDEN,
        n_shared_blocks = SHARED_BLOCKS,
        n_gcn_blocks = GCN_BLOCKS,
        n_scalar_features = len(data_cfg.scalar_columns),
        n_node_features = len(data_cfg.node_columns),
        n_outputs = len(data_cfg.target_columns),
    )
    train_cfg = TrainConfig(
        # early stopping wont trigger during smoke test
        early_stopping_patience = 10**9 if SMOKE else PATIENCE,
    )
    return train(data_cfg, model_cfg, train_cfg)
 
 
if __name__ == "__main__":
    main()