# libraries

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from data import CropBox, DataConfig, ScalerBundle, build_splits
from metrics import (NMAPEAccumulator, RelativeErrorAccumulator, RelativeErrorConfig,
                     reference_values)
from model import GCNSurrogate, GCNSurrogateConfig, count_parameters

# settings

RUN_DIR = Path(r"C:\Users\26664984\Documents\Masters\Model_training\iter1")
H5_PATH = r"C:\Users\26664984\Documents\Masters\hdf5_training_data\hydrofoil.h5"

SPLIT = "test" # train, val or test
THRESHOLDS = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1]
MODE = "mask"  # floor

RHO = 1025.0 # working fluid density for the NMAPE reference

EXPORT_CASES = 3      
EXPORT_DIR = RUN_DIR / "fields"

PARITY_ALL = True # predicted vs actual for every test case
PARITY_STRIDE = 10  

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_run() -> tuple[GCNSurrogate, DataConfig, dict]:
    "Rebuild the model and the data config from the run directory."
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
    )

    model_cfg = GCNSurrogateConfig(**saved["model"])
    model = GCNSurrogate(model_cfg).to(DEVICE)

    ckpt = torch.load(RUN_DIR / "best.pt", map_location = DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[eval] loaded epoch {ckpt['epoch']} (val loss {ckpt['val_loss']:.6e}), "
          f"{count_parameters(model):,} parameters on {DEVICE}")
    return model, data_cfg, saved


@torch.no_grad()
def relative_errors(model, loader, scalers: ScalerBundle, cfg: RelativeErrorConfig) -> dict:
    "Two passes: field scale, then the errors themselves."
    acc = RelativeErrorAccumulator(scalers.target_columns, cfg)
    for batch in loader:
        acc.update_scale(scalers.target.inverse_transform(batch.y.to(DEVICE)))
    acc.lock_scale()

    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch.x, batch.edge_index, batch.scalars, batch.batch)
        # both sides denormalized
        acc.update(
            scalers.target.inverse_transform(pred),
            scalers.target.inverse_transform(batch.y),
        )
    return acc.result()


@torch.no_grad()
def nmape(model, loader, scalers: ScalerBundle) -> dict:
    "Normalised by a per-case reference"
    acc = NMAPEAccumulator(scalers.target_columns)
    v_index = list(scalers.scalar_columns).index("V_in")

    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch.x, batch.edge_index, batch.scalars, batch.batch)
        y_pred = scalers.target.inverse_transform(pred)
        y_true = scalers.target.inverse_transform(batch.y)
        scalars_phys = scalers.scalar.inverse_transform(batch.scalars.cpu().numpy())

        # one graph at a time, since the reference differs per case
        offset = 0
        for g in range(scalars_phys.shape[0]):
            n = int((batch.batch == g).sum().item())
            ref = reference_values(float(scalars_phys[g, v_index]),
                                   scalers.target_columns, rho = RHO)
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
        y_pred = scalers.target.inverse_transform(pred).cpu().numpy()
        y_true = scalers.target.inverse_transform(batch.y).cpu().numpy()
        scalars_phys = scalers.scalar.inverse_transform(batch.scalars.cpu().numpy())[0]

        out = EXPORT_DIR / f"dp{dp_id}.npz"
        np.savez_compressed(
            out,
            x = xy[:, 0], y = xy[:, 1],
            pred = y_pred, true = y_true,
            scalars = scalars_phys,
            target_columns = np.array(scalers.target_columns, dtype = object),
            scalar_columns = np.array(scalers.scalar_columns, dtype = object),
        )
        err = np.abs(y_pred - y_true)
        print(f"[eval] dp {dp_id}: {xy.shape[0]:,} nodes, max |error| "
              + ", ".join(f"{n}={err[:, k].max():.4g}"
                          for k, n in enumerate(scalers.target_columns))
              + f" -> {out.name}")


@torch.no_grad()
def export_parity(model, loader, scalers: ScalerBundle, path: Path, stride: int) -> None:
    "Predicted and actual for every case in the split. Only the target values and the camber are kept"
    preds, trues, cambers, dp_ids = [], [], [], []
    m_index = list(scalers.scalar_columns).index("m")

    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch.x, batch.edge_index, batch.scalars, batch.batch)
        p = scalers.target.inverse_transform(pred).cpu().numpy()[::stride]
        t = scalers.target.inverse_transform(batch.y).cpu().numpy()[::stride]
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
    model, data_cfg, saved = load_run()

    train_ds, val_ds, test_ds, scalers, store = build_splits(data_cfg, verbose = False)
    datasets = {"train": train_ds, "val": val_ds, "test": test_ds}
    dataset = datasets[SPLIT]
    loader = DataLoader(dataset, batch_size = 1)
    print(f"[eval] {SPLIT} split: {len(dataset)} cases")

    # threshold sweep.
    rows = []
    for thr in THRESHOLDS:
        cfg = RelativeErrorConfig(mode = MODE, threshold = thr)
        result = relative_errors(model, loader, scalers, cfg)
        rows.append((thr, result))

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

    print()
    if EXPORT_CASES:
        export_fields(model, dataset, scalers, EXPORT_CASES)

    if PARITY_ALL:
        EXPORT_DIR.mkdir(parents = True, exist_ok = True)
        export_parity(model, loader, scalers,
                      RUN_DIR / f"parity_{SPLIT}.npz", PARITY_STRIDE)

    store.close()


if __name__ == "__main__":
    main()