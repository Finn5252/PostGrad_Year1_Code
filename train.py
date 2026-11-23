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

from data import CropBox, DataConfig, build_splits
from evaluate import relative_errors
from metrics import MeanAccumulator, RelativeErrorConfig
from model import GCNSurrogate, GCNSurrogateConfig, count_parameters
from regularisation import REG_MODES, laplacian_penalty
from settings import H5_PATH, OUT_DIR, RHO
from surface import (SURFACE_LOSSES, WEIGHTING_MODES, CoVWeighting, DragLoss, FixedWeighting, PanelCache,
                     SurfaceCpLoss)

# settings

CROP = (-0.2, 1.50, 0.80, 1.4) # xmin, xmax, ymin, ymax
KNN_K = 4 # k nearest neighbours for gcn
CACHE_DIR = "cache" # KNN graphs and surface panels

TARGET_NORMALISATION = "freestream" # "none" or "freestream" (Cp and V / V_in, then min-max)

EPOCHS = 1000
BATCH_SIZE = 1 # cases per optimiser step
LR = 1e-3 # initial learning rate
LR_SCHEDULE = "cosine" # "cosine" or "step"
LR_MIN = 1e-5 # learning rate reached at the final epoch (cosine only)
PATIENCE = 200 # how many epochs validation loss must see no improvement for before early stopping
SEED = 0 # for replicability

HIDDEN = 256 # neurons per layer
SHARED_BLOCKS = 4 # shared GCN/FC blocks
GCN_BLOCKS = 3 # residual GCN blocks after the shared ones
CONV = "gatv2" # "gcn", "gat" or "gatv2"
HEADS = 4 # attention heads

# smoothness regularization
REG_K = None # neighbours in the gaussian graph for the smoothing loss, None = no graph
REG_LAMBDA = 0 # weight of the smoothing loss, 0 = not trained on
REG_MODE = "error" # "error" penalises roughness of prediction - target, "prediction" of the prediction itself

# additional loss
SURFACE_LOSS = "cp" # None, "drag" or "cp"
LOSS_WEIGHTING = "fixed" # "fixed" or "cov"
SURFACE_LAMBDA = 1.0 # weight of the surface term against the field term
FIELD_LOSS_REF = 4.7e-6 # typical field loss size
DRAG_LOSS_REF = 2e-4 # typical drag loss size
CP_LOSS_REF = 4.0e-5 # typical cp loss size
COV_WINDOW = None # optimiser steps averaged into each CoV observation; None = one epoch
COV_WARMUP = None # optimiser steps at equal weights before CoV takes over; None = two windows

ZERO_HANDLING = "mask" # "mask" or "floor": how near-zero targets are treated in the relative errors
ZERO_THRESHOLD = 1e-3 # as a fraction of the field RMS

LOG_TEST_EACH_EPOCH = False 

SMOKE = False # smoke test


# configuration

@dataclass
class TrainConfig:
    out_dir: str = OUT_DIR
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    lr: float = LR
    betas: tuple[float, float] = (0.9, 0.999)  # Adam momentum and scaling decay
    weight_decay: float = 0.0 # L2 penalty on the weights, off
    lr_decay_rate: float = 0.9 # step schedule only: the learning rate is multiplied by this
    lr_decay_every_epochs: int = 200 # step schedule only: epochs between decays
    lr_schedule: str = LR_SCHEDULE
    lr_min: float = LR_MIN
    early_stopping_patience: int = PATIENCE
    early_stopping_min_delta: float = 0.0 # smallest drop in validation loss that counts as an improvement

    reg_lambda: float = REG_LAMBDA
    reg_mode: str = REG_MODE

    surface_loss: Optional[str] = SURFACE_LOSS
    loss_weighting: str = LOSS_WEIGHTING
    surface_lambda: float = SURFACE_LAMBDA
    field_loss_ref: float = FIELD_LOSS_REF
    drag_loss_ref: float = DRAG_LOSS_REF
    cp_loss_ref: float = CP_LOSS_REF
    cov_window: Optional[int] = COV_WINDOW
    cov_warmup: Optional[int] = COV_WARMUP

    seed: int = SEED
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
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
        if self.lr_schedule not in ("cosine", "step"):
            raise ValueError(f"lr_schedule must be 'cosine' or 'step', got {self.lr_schedule!r}")
        if self.surface_loss not in SURFACE_LOSSES:
            raise ValueError(f"surface_loss must be one of {SURFACE_LOSSES}, got {self.surface_loss!r}")
        if self.loss_weighting not in WEIGHTING_MODES:
            raise ValueError(f"loss_weighting must be one of {WEIGHTING_MODES}, got {self.loss_weighting!r}")
        if self.surface_lambda < 0:
            raise ValueError("surface_lambda must be >= 0")
        if self.field_loss_ref <= 0 or self.drag_loss_ref <= 0 or self.cp_loss_ref <= 0:
            raise ValueError("field_loss_ref, drag_loss_ref and cp_loss_ref must be > 0")
        if self.cov_window is not None and self.cov_window < 1:
            raise ValueError("cov_window must be >= 1, or None for one epoch")
        if self.cov_warmup is not None and self.cov_warmup < 0:
            raise ValueError("cov_warmup must be >= 0, or None for two windows")

    @property
    def surface_loss_ref(self) -> float:
        return self.drag_loss_ref if self.surface_loss == "drag" else self.cp_loss_ref

def set_seed(seed: int) -> None:
    "Four separate generators, because each library keeps its own"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Epoch helpers

def _forward(model: GCNSurrogate, batch, device: str):
    # the whole batch is returned on the device, since the losses need more of it than the prediction
    batch = batch.to(device)
    pred = model(batch.x, batch.edge_index, batch.scalars, batch.batch)
    return pred, batch

def _losses(pred, batch, reg_lambda: float, reg_mode: str, surface_loss = None):
    "Data term, smoothness term and surface term"
    data_loss = 0.5 * nn.functional.mse_loss(pred, batch.y)
    reg = None
    if "reg_edge_index" in batch:
        args = (pred, batch.y, batch.reg_edge_index, batch.reg_edge_weight, reg_mode)
        if reg_lambda > 0:
            reg = laplacian_penalty(*args)
        else:
            # lambda = 0: the term is still measured
            with torch.no_grad():
                reg = laplacian_penalty(*args)
    surface = surface_loss(pred, batch) if surface_loss is not None else None
    return data_loss, reg, surface

def _combine(data_loss, reg, surface, reg_lambda: float, weights: tuple[float, float]):
    "w_field * data + reg_lambda * reg + w_surface * surface"
    total = weights[0] * data_loss
    if reg is not None and reg_lambda > 0:
        total = total + reg_lambda * reg
    if surface is not None:
        total = total + weights[1] * surface
    return total

def run_epoch(
        model, loader, device, reg_lambda: float, reg_mode: str, optimizer = None,
        surface_loss = None, weighting = None,
) -> dict:
    "One complete pass over a dataset, returning the case-averaged total, data, smoothness and surface terms."
    training = optimizer is not None
    model.train(training)
    acc = {name: MeanAccumulator() for name in ("loss", "data", "reg", "surface", "w_field", "w_surface")}
    has_reg = False
    reference = weighting.reference if weighting is not None else (1.0, 0.0)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            pred, batch = _forward(model, batch, device)
            data_loss, reg, surface = _losses(pred, batch, reg_lambda, reg_mode, surface_loss)
            # the logged loss always uses the reference weights, so epochs stay comparable while the cov weights move
            loss = _combine(data_loss, reg, surface, reg_lambda, reference)
            if training:
                w = weighting.weights() if weighting is not None else reference
                # clear the previous gradients, adjust weights
                optimizer.zero_grad(set_to_none = True)
                _combine(data_loss, reg, surface, reg_lambda, w).backward()
                optimizer.step()
                if weighting is not None:
                    weighting.observe(data_loss.item(), surface.item())
                acc["w_field"].update(w[0])
                acc["w_surface"].update(w[1])
            n_graphs = int(batch.num_graphs)
            # .item() keeps the number only, not the graph of operations behind it
            acc["loss"].update(loss.item(), n = n_graphs)
            acc["data"].update(data_loss.item(), n = n_graphs)
            if reg is not None:
                acc["reg"].update(reg.item(), n = n_graphs)
                has_reg = True
            if surface is not None:
                acc["surface"].update(surface.item(), n = n_graphs)
    out = {
        "loss": acc["loss"].mean,
        "data": acc["data"].mean,
        "reg": acc["reg"].mean if has_reg else float("nan"),
        "surface": acc["surface"].mean,
    }
    if training:
        out["w_field"], out["w_surface"] = acc["w_field"].mean, acc["w_surface"].mean
    return out

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
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}

    surface_loss, weighting = None, None
    if train_cfg.surface_loss is not None:
        panel_cache = PanelCache(data_cfg, store)
        panel_cache.build_or_load()
        for ds in (train_ds, val_ds, test_ds):
            ds.panel_cache = panel_cache
        surface_loss = (DragLoss if train_cfg.surface_loss == "drag" else SurfaceCpLoss)(scalers)
        # each loss over its reference, all times field_loss_ref so the field term keeps its usual scale
        scale = train_cfg.field_loss_ref / train_cfg.surface_loss_ref
        reference = (1.0, train_cfg.surface_lambda * scale)
        if train_cfg.loss_weighting == "fixed":
            weighting = FixedWeighting(reference)
        else:
            window = train_cfg.cov_window or len(train_loader)
            warmup = 2 * window if train_cfg.cov_warmup is None else train_cfg.cov_warmup
            weighting = CoVWeighting(reference, scale, window, warmup)

    model = GCNSurrogate(model_cfg).to(device)
    print(f"[train] {count_parameters(model):,} trainable parameters on {device}")
    if data_cfg.reg_k is not None:
        print(f"[train] smoothness regularization: lambda={lam}, mode={mode!r}, reg_k={data_cfg.reg_k}")
    if surface_loss is not None:
        print(f"[train] surface loss: {train_cfg.surface_loss}, weighting: {weighting.summary()}")
    # names the surface term in the logs and history.csv: "drag", "cp", or "surface" when there is none
    label = train_cfg.surface_loss or "surface"

    def terms(res: dict, surface_key: str = "surface", digits: int = 3) -> str:
        "The loss terms that are switched on, for the log lines"
        text = f"data={res['data']:.{digits}e}"
        if data_cfg.reg_k is not None:
            text += f" reg={res['reg']:.{digits}e}"
        if surface_loss is not None:
            text += f" {label}={res[surface_key]:.{digits}e}"
        return text

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = train_cfg.lr,
        betas = train_cfg.betas,
        weight_decay = train_cfg.weight_decay,
    )

    if train_cfg.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max = train_cfg.epochs, eta_min = train_cfg.lr_min
        )
    else:
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
                "weighting": weighting.summary() if weighting is not None else None,
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

    fieldnames = (
        ["epoch", "lr"]
        + [f"{split}_{part}" for split in loaders for part in ("loss", "data", "reg")]
        + [f"{split}_{label}" for split in loaders]
        + ["w_field", f"w_{label}"]
    )
    # stands in for the test results when the test split is not run every epoch
    nan_row = {"loss": float("nan"), "data": float("nan"), "reg": float("nan"), "surface": float("nan")}

    with history_path.open("w", newline = "") as fh:
        writer = csv.DictWriter(fh, fieldnames = fieldnames)
        writer.writeheader()

        for epoch in range(1, train_cfg.epochs + 1):
            lr_now = optimizer.param_groups[0]["lr"]
            tr = run_epoch(model, train_loader, device, lam, mode, optimizer, surface_loss, weighting)
            va = run_epoch(model, val_loader, device, lam, mode, None, surface_loss, weighting)
            te = (
                run_epoch(model, test_loader, device, lam, mode, None, surface_loss, weighting)
                if train_cfg.log_test_each_epoch
                else nan_row
            )
            scheduler.step()

            row = {"epoch": epoch, "lr": lr_now, "w_field": tr.pop("w_field"), f"w_{label}": tr.pop("w_surface")}
            for split, res in (("train", tr), ("val", va), ("test", te)):
                for part, value in res.items():
                    row[f"{split}_{label if part == 'surface' else part}"] = value
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
                w_text = f"w=({row['w_field']:.3g}, {row[f'w_{label}']:.3g}) " if surface_loss is not None else ""
                print(
                    f"[{epoch:>4}/{train_cfg.epochs}] lr={lr_now:.2e} "
                    f"train={tr['loss']:.6e} val={va['loss']:.6e} "
                    f"(val {terms(va)}) {w_text}"
                    f"best_val={best_val:.6e}@{best_epoch}"
                )

            if epochs_since_improvement >= train_cfg.early_stopping_patience:
                print(
                    f"[train] early stopping at epoch {epoch}: no validation improvement "
                    f"for {train_cfg.early_stopping_patience} epochs"
                )
                break

    # the best checkpoint is reloaded for a single, final evaluation
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location = device)["model_state"])
    model.eval()

    results = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "wall_time_s": time.time() - t0,
        "n_parameters": count_parameters(model),
        "reg": {"lambda": lam, "mode": mode, "k": data_cfg.reg_k},
        "surface_loss": train_cfg.surface_loss,
        "weighting": weighting.summary() if weighting is not None else None,
        "n_cases": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "loss": {
            split: run_epoch(model, loader, device, lam, mode, None, surface_loss, weighting)
            for split, loader in loaders.items()
        },
        "relative_errors": {
            split: relative_errors(model, loader, scalers, train_cfg.relative_error, device)
            for split, loader in loaders.items()
        },
    }
    for res in results["loss"].values():
        res[label] = res.pop("surface")
    (out / "results.json").write_text(json.dumps(results, indent = 2))

    print(f"\n[train] best epoch {best_epoch} (val loss {best_val:.6e})")
    t = results["loss"]["test"]
    print(f"[train] final test (evaluated once): loss={t['loss']:.6e} {terms(t, label, digits = 6)}")
    for split in loaders:
        print(f"[train] relative errors ({split}):")
        for name, r in results["relative_errors"][split].items():
            print(
                f"    {name:<20} L1={r['L1']:.4f}  L2={r['L2']:.4f}  "
                f"excluded={r['excluded_fraction']:.3%}  ({r['handling']})"
            )
    store.close()
    return results

def _jsonable(obj):
    "Tuples become lists, recursively (JSON has no tuple type)"
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj

def configs() -> tuple[DataConfig, GCNSurrogateConfig, TrainConfig]:
    "The settings above as config objects"
    data_cfg = DataConfig(
        h5_path = H5_PATH,
        # *CROP unpacks the four values into four separate arguments
        crop = CropBox(*CROP) if CROP else None,
        knn_k = KNN_K,
        reg_k = REG_K,
        cache_dir = CACHE_DIR,
        target_normalisation = TARGET_NORMALISATION,
        rho = RHO,
        split_seed = SEED,
    )
    model_cfg = GCNSurrogateConfig(
        hidden = HIDDEN,
        n_shared_blocks = SHARED_BLOCKS,
        n_gcn_blocks = GCN_BLOCKS,
        n_scalar_features = len(data_cfg.scalar_columns),
        n_node_features = len(data_cfg.node_columns),
        n_outputs = len(data_cfg.target_columns),
        conv = CONV,
        heads = HEADS,
    )
    train_cfg = TrainConfig(
        # a patience this large never triggers, so a smoke test runs all its epochs
        early_stopping_patience = 10**9 if SMOKE else PATIENCE,
    )
    return data_cfg, model_cfg, train_cfg

def main() -> dict:
    return train(*configs())


if __name__ == "__main__":
    main()
