"""Load a trained checkpoint and evaluate it without retraining.

Two things:

  * sweeps the near-zero threshold used by the relative L1/L2 metrics, so you can see
    how much of the reported error comes from nodes whose true value sits close to zero
    (velocity at the stagnation point, pressure where it crosses zero);
  * writes denormalized predicted and actual fields for chosen cases to .npz, ready for
    contour plotting.

The DataConfig here must match the one the run was trained with, or the splits will
differ and the "test" set will not be the same cases. The values are read back from the
run's config.json where possible.

Run:  python evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from data import CropBox, DataConfig, ScalerBundle, build_splits
from metrics import RelativeErrorAccumulator, RelativeErrorConfig
from model import GCNSurrogate, GCNSurrogateConfig, count_parameters

# settings

RUN_DIR = Path(r"C:\Users\26664984\Documents\Masters\Model_training\iter1")
H5_PATH = r"C:\Users\26664984\Documents\Masters\hdf5_training_data\hydrofoil.h5"

SPLIT = "test"                  # "train", "val" or "test"
THRESHOLDS = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1]
MODE = "mask"                   # "mask" or "floor"

EXPORT_CASES = 3                # write this many cases' fields to .npz; 0 to skip
EXPORT_DIR = RUN_DIR / "fields"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_run() -> tuple[GCNSurrogate, DataConfig, dict]:
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
    "Two passes: field scale, then the errors themselves"
    acc = RelativeErrorAccumulator(scalers.target_columns, cfg)
    for batch in loader:
        acc.update_scale(scalers.target.inverse_transform(batch.y.to(DEVICE)))
    acc.lock_scale()

    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch.x, batch.edge_index, batch.scalars, batch.batch)
        acc.update(
            scalers.target.inverse_transform(pred),
            scalers.target.inverse_transform(batch.y),
        )
    return acc.result()


@torch.no_grad()
def export_fields(model, dataset, scalers: ScalerBundle, n_cases: int) -> None:
    "Write denormalized coordinates, predictions and truth for the first n cases"
    EXPORT_DIR.mkdir(parents = True, exist_ok = True)
    for j in range(min(n_cases, len(dataset))):
        data = dataset[j]
        dp_id = int(data.dp_id.item())
        batch = data.to(DEVICE)
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


def main() -> None:
    model, data_cfg, saved = load_run()

    train_ds, val_ds, test_ds, scalers, store = build_splits(data_cfg, verbose = False)
    datasets = {"train": train_ds, "val": val_ds, "test": test_ds}
    dataset = datasets[SPLIT]
    loader = DataLoader(dataset, batch_size = 1)
    print(f"[eval] {SPLIT} split: {len(dataset)} cases")

    # threshold sweep
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
    print("If L2 falls sharply as the threshold rises while L1 barely moves, the L2")
    print("figure is dominated by a few near-zero denominators rather than by the")
    print("model's accuracy over the field.")

    (RUN_DIR / f"threshold_sweep_{SPLIT}.json").write_text(
        json.dumps({f"{thr:g}": r for thr, r in rows}, indent = 2)
    )

    if EXPORT_CASES:
        print()
        export_fields(model, dataset, scalers, EXPORT_CASES)

    store.close()


if __name__ == "__main__":
    main()