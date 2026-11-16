# libraries

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from data import (CropBox, DataConfig, ScalerBundle, build_splits, freestream_reference,
                  targets_to_physical)
from metrics import NMAPEAccumulator, RelativeErrorAccumulator, RelativeErrorConfig
from model import GCNSurrogate, GCNSurrogateConfig, count_parameters
from settings import H5_PATH, RHO, RUN_DIR

# settings

SPLIT = "test" # "train", "val" or "test"
THRESHOLDS = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1] # near-zero thresholds swept for the relative errors
MODE = "mask" # "mask" or "floor", see RelativeErrorConfig

EXPORT_CASES = 3 # full fields written for the first few cases, for plot.py. 0 turns it off
EXPORT_DIR = RUN_DIR / "fields"

PARITY_ALL = True # predicted vs actual for every case in the split
PARITY_STRIDE = 1 # keep every n-th node in the parity file

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_run(tag: str = "eval") -> tuple[GCNSurrogate, DataConfig]:
    "Rebuild the model and the data config from the run directory"
    cfg_path = RUN_DIR / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"no config.json in {RUN_DIR}")
    saved = json.loads(cfg_path.read_text())

    data_saved = saved["data"]
    crop = data_saved.get("crop")
    data_cfg = DataConfig(
        h5_path = H5_PATH,
        node_columns = tuple(data_saved["node_columns"]),
        target_columns = tuple(data_saved["target_columns"]),
        scalar_columns = tuple(data_saved["scalar_columns"]),
        crop = CropBox(**crop) if crop else None,
        knn_k = data_saved["knn_k"],
        cache_dir = data_saved["cache_dir"],
        split_fractions = tuple(data_saved["split_fractions"]),
        split_seed = data_saved["split_seed"],
        # the run's own setting, not train.py's. Older runs were not normalised
        target_normalisation = data_saved.get("target_normalisation", "none"),
        rho = data_saved.get("rho", RHO),
    )
    check_scalers(data_cfg, RUN_DIR)

    model_cfg = GCNSurrogateConfig(**saved["model"])
    model = GCNSurrogate(model_cfg).to(DEVICE)

    ckpt = torch.load(RUN_DIR / "best.pt", map_location = DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[{tag}] loaded epoch {ckpt['epoch']} (val loss {ckpt['val_loss']:.6e}), "
          f"{count_parameters(model):,} parameters on {DEVICE}")
    return model, data_cfg


def check_scalers(data_cfg: DataConfig, run_dir: Path) -> None:
    "The saved scalers.json must agree with config.json on the normalisation"
    path = run_dir / "scalers.json"
    if not path.exists():
        return
    saved = ScalerBundle.load(path)
    if (saved.target_normalisation, saved.rho) != (data_cfg.target_normalisation, data_cfg.rho):
        raise ValueError(
            f"{path.name} has target_normalisation={saved.target_normalisation!r}, rho={saved.rho} "
            f"but config.json has {data_cfg.target_normalisation!r}, rho={data_cfg.rho}"
        )


@torch.no_grad()
def relative_errors(model, loader, scalers: ScalerBundle, cfg: RelativeErrorConfig, device: str) -> dict:
    "Relative L1/L2 per target field, in Pa and m/s. Two passes: field scale, then the errors themselves"
    model.eval()
    acc = RelativeErrorAccumulator(scalers.target_columns, cfg)
    for batch in loader:
        acc.update_scale(targets_to_physical(batch.y.to(device), batch, scalers))
    acc.lock_scale()

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.scalars, batch.batch)
        acc.update(
            targets_to_physical(pred, batch, scalers),
            targets_to_physical(batch.y, batch, scalers),
        )
    return acc.result()


@torch.no_grad()
def nmape(model, loader, scalers: ScalerBundle) -> dict:
    "Error as a percentage of a per-case reference: V_in for velocity, the dynamic pressure for pressure"
    acc = NMAPEAccumulator(scalers.target_columns)
    v_index = list(scalers.scalar_columns).index("V_in")

    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch.x, batch.edge_index, batch.scalars, batch.batch)
        y_pred = targets_to_physical(pred, batch, scalers)
        y_true = targets_to_physical(batch.y, batch, scalers)
        scalars_phys = scalers.scalar.inverse_transform(batch.scalars.cpu().numpy())

        # one graph at a time, since the reference differs per case
        offset = 0
        for g in range(scalars_phys.shape[0]):
            n = int((batch.batch == g).sum().item())
            v_in = float(scalars_phys[g, v_index])
            ref = freestream_reference(v_in, scalers.target_columns, scalers.rho)[0]
            acc.update(y_pred[offset:offset + n], y_true[offset:offset + n], ref)
            offset += n
    return acc.result()


@torch.no_grad()
def export_fields(model, dataset, scalers: ScalerBundle, n_cases: int) -> None:
    "Write denormalized coordinates, predictions and truth for the first n cases"
    EXPORT_DIR.mkdir(parents = True, exist_ok = True)
    for j in range(min(n_cases, len(dataset))):
        data = dataset[j]
        dp_id = int(data.dp_id.item())
        batch = data.to(DEVICE)
        # batch = None: a single graph, so every node belongs to case 0
        pred = model(batch.x, batch.edge_index, batch.scalars, None)

        xy = scalers.node.inverse_transform(batch.x.cpu().numpy())[:, :2]
        y_pred = targets_to_physical(pred, batch, scalers).cpu().numpy()
        y_true = targets_to_physical(batch.y, batch, scalers).cpu().numpy()
        scalars_phys = scalers.scalar.inverse_transform(batch.scalars.cpu().numpy())[0]

        np.savez_compressed(
            EXPORT_DIR / f"dp{dp_id}.npz",
            x = xy[:, 0], y = xy[:, 1],
            pred = y_pred, true = y_true,
            scalars = scalars_phys,
            # dtype = object stores the names as strings, which plot.py reads back with allow_pickle
            target_columns = np.array(scalers.target_columns, dtype = object),
            scalar_columns = np.array(scalers.scalar_columns, dtype = object),
        )


@torch.no_grad()
def export_parity(model, loader, scalers: ScalerBundle, path: Path, stride: int) -> None:
    "Predicted and actual for every case in the split. Only the target values and the camber are kept"
    preds, trues, cambers, dp_ids = [], [], [], []
    m_index = list(scalers.scalar_columns).index("m")

    # assumes a batch size of 1, so each batch is one case
    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch.x, batch.edge_index, batch.scalars, batch.batch)
        p = targets_to_physical(pred, batch, scalers).cpu().numpy()[::stride]
        t = targets_to_physical(batch.y, batch, scalers).cpu().numpy()[::stride]
        s = scalers.scalar.inverse_transform(batch.scalars.cpu().numpy())

        preds.append(p.astype(np.float32))
        trues.append(t.astype(np.float32))
        # one camber value repeated per node
        cambers.append(np.full(p.shape[0], s[0, m_index], dtype = np.float32))
        dp_ids.append(int(batch.dp_id[0].item()))

    pred_all = np.concatenate(preds)
    np.savez_compressed(
        path,
        pred = pred_all,
        true = np.concatenate(trues),
        camber = np.concatenate(cambers),
        dp_ids = np.asarray(dp_ids, dtype = np.int64),
        stride = stride,
        target_columns = np.array(scalers.target_columns, dtype = object),
    )
    print(f"[eval] parity: {len(dp_ids)} cases, {pred_all.shape[0]:,} points per field "
          f"(every {stride}th node) -> {path.name}")


def main() -> None:
    model, data_cfg = load_run()

    train_ds, val_ds, test_ds, scalers, store = build_splits(data_cfg, verbose = False)
    dataset = {"train": train_ds, "val": val_ds, "test": test_ds}[SPLIT]
    loader = DataLoader(dataset, batch_size = 1)
    print(f"[eval] {SPLIT} split: {len(dataset)} cases")

    # threshold sweep, to show how much the relative errors depend on the near-zero cut-off
    rows = []
    for thr in THRESHOLDS:
        cfg = RelativeErrorConfig(mode = MODE, threshold = thr)
        rows.append((thr, relative_errors(model, loader, scalers, cfg, DEVICE)))

    names = list(scalers.target_columns)
    width = max(len(n) for n in names)
    print("\n" + "=" * 78)
    print(f"near-zero threshold sweep ({MODE}, threshold x field RMS), {SPLIT} split")
    print("=" * 78)
    header = f"{'threshold':>10}"
    for n in names:
        header += f" | {n[:width]:>{width}} L1{'':>4}L2{'':>5}excl"
    print(header)
    for thr, result in rows:
        line = f"{thr:>10.0e}"
        for n in names:
            r = result[n]
            line += (f" | {r['L1']:>{width}.4f}  {r['L2']:>7.4f}  "
                     f"{r['excluded_fraction']:>6.2%}")
        print(line)
    print("=" * 78)

    # the same model measured without a threshold
    nm = nmape(model, loader, scalers)
    print("\n" + "=" * 78)
    print(f"NMAPE ({SPLIT} split)")
    print("=" * 78)
    for name, r in nm.items():
        print(f"  {name:<20} NMAPE = {r['NMAPE']:>7.3f}%   "
              f"worst node = {r['max_percent']:.2f}%")
    print("=" * 78)

    (RUN_DIR / f"nmape_{SPLIT}.json").write_text(json.dumps(nm, indent = 2))
    (RUN_DIR / f"threshold_sweep_{SPLIT}.json").write_text(
        json.dumps({f"{thr:g}": r for thr, r in rows}, indent = 2)
    )

    if EXPORT_CASES:
        export_fields(model, dataset, scalers, EXPORT_CASES)
    if PARITY_ALL:
        export_parity(model, loader, scalers, RUN_DIR / f"parity_{SPLIT}.npz", PARITY_STRIDE)

    store.close()


if __name__ == "__main__":
    main()
