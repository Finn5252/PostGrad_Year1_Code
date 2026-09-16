# packages

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd

# settings

CSV_DIR = Path(r"C:\exports")
CSV_GLOB = "*.csv"
PARAM_TABLE = Path(r"C:\Users\26664984\Documents\Masters\Simulations\parameter set.csv")
OUT_PATH = Path(r"C:\Users\26664984\Documents\Masters\hdf5_training_data\hydrofoil.h5")

LIMIT: Optional[int] = None # pilot processing

DP_ID_FROM_FILENAME = r"(\d+)"

PARAM_SEP = ","
PARAM_DECIMAL = "."

MIN_ROWS_PER_CASE = 1000
EXCLUDE_DP_IDS = {0}    # DP 0 is the Workbench base design point, not part of the DOE
PROGRESS_EVERY = 25

CSV_HEADER = ["cellnumber", "x-coordinate", "y-coordinate", "velocity-magnitude", "pressure", "cell-volume", "y-coordinate", "x-coordinate",]
USE_COLS = [0, 1, 2, 3, 4, 5]

NODE_COLUMNS = ["x", "y", "cell_volume"]
TARGET_COLUMNS = ["static_pressure", "velocity_magnitude"]
SCALAR_COLUMNS = ["m", "p", "t", "AoA", "V_in"]

PARAM_MAP = {"P1": "m", "P2": "p", "P3": "t", "P4": "AoA", "P18": "V_in"}

def fail(message: str) -> None:
    raise RuntimeError(message)

#reading

def read_header(path: Path) -> list[str]:
    with path.open("r", encoding = "utf-8-sig") as fh:
        line = fh.readline()
    return [name.strip() for name in line.rstrip("\n").split(",")]
 
 
def read_case(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    "Return (node_data (n,3), targets (n,2), cell_ids (n,))"
    frame = pd.read_csv(
        path,
        header = 0,
        usecols = USE_COLS,
        names = [f"c{i}" for i in range(len(CSV_HEADER))],
        dtype = np.float64,
        skipinitialspace = True,
    )
    if frame.empty:
        fail(f"{path.name} has a header but no data rows")
 
    node = np.stack([frame["c1"], frame["c2"], frame["c5"]], axis = 1)      # x, y, volume
    targets = np.stack([frame["c4"], frame["c3"]], axis = 1)               # pressure, velocity
    cell_ids = frame["c0"].to_numpy()
 
    for name, arr in (("node_data", node), ("targets", targets)):
        if not np.all(np.isfinite(arr)):
            rows = np.unique(np.nonzero(~np.isfinite(arr))[0])[:5]
            fail(f"{path.name}: non-finite values in {name}, first rows {rows.tolist()}")
    if np.any(cell_ids != np.floor(cell_ids)):
        fail(f"{path.name}: cell numbers are not integral -- wrong column or separator?")
 
    return node.astype(np.float32), targets.astype(np.float32), cell_ids.astype(np.int64)
 
 
def load_param_table() -> dict[int, np.ndarray]:
    "Return dp_id -> [m, p, t, AoA, V_in]"
    if not PARAM_TABLE.exists():
        fail(f"parameter table not found: {PARAM_TABLE}")
 
    frame = pd.read_csv(
        PARAM_TABLE,
        sep = PARAM_SEP,
        decimal = PARAM_DECIMAL,
        header = 0,
        comment = "#",
        encoding = "utf-8-sig",
        skipinitialspace = True,
    )
    frame.columns = [str(c).strip() for c in frame.columns]   
    columns = list(frame.columns)
 
    resolved = {}
    for token, name in PARAM_MAP.items():
        hits = [c for c in columns if re.match(rf"^{token}(?![0-9])", c)]
        if len(hits) != 1:
            fail(f"parameter {token} ({name}) matched {hits} in {PARAM_TABLE.name}")
        resolved[name] = hits[0]
    print("[pack] mapped " + ", ".join(f"{k} <- {v!r}" for k, v in resolved.items()))
 
    ids = []
    for raw in frame[columns[0]].astype(str).str.strip():
        found = re.findall(r"\d+", raw)
        if not found:
            fail(f"cannot read a design point number from {raw!r}")
        ids.append(int(found[-1]))
    if len(set(ids)) != len(ids):
        fail("duplicate design point IDs in the parameter table")
 
    values = np.stack(
        [pd.to_numeric(frame[resolved[n]], errors = "coerce").to_numpy() for n in SCALAR_COLUMNS],
        axis = 1,
    )
    if not np.all(np.isfinite(values)):
        bad = np.unique(np.nonzero(~np.isfinite(values))[1])
        fail(
            f"non-finite parameter values in columns {[SCALAR_COLUMNS[j] for j in bad]}. "
            f"Check PARAM_SEP={PARAM_SEP!r} and PARAM_DECIMAL={PARAM_DECIMAL!r}."
        )
 
    table = {i: values[k].astype(np.float32) for k, i in enumerate(ids)}
    for i in EXCLUDE_DP_IDS:
        table.pop(i, None)
    return table 
 
def discover_cases() -> dict[int, Path]:
    if not CSV_DIR.is_dir():
        fail(f"CSV directory not found: {CSV_DIR}")
    paths = sorted(CSV_DIR.glob(CSV_GLOB))
    if not paths:
        fail(f"no files matching {CSV_GLOB!r} in {CSV_DIR}")
 
    mapping: dict[int, Path] = {}
    for path in paths:
        found = re.findall(DP_ID_FROM_FILENAME, path.stem)
        if not found:
            fail(f"no design point ID in {path.name!r} using {DP_ID_FROM_FILENAME!r}")
        dp_id = int(found[-1])
        if dp_id in mapping:
            fail(f"dp {dp_id} extracted from both {mapping[dp_id].name} and {path.name}")
        mapping[dp_id] = path
    for i in EXCLUDE_DP_IDS:
        mapping.pop(i, None)    
    return mapping

# writing
 
def verify_written_file(path: Path, param_table: dict[int, np.ndarray]) -> None:
    "Re-read the finished file and re-check the join from its own contents"
    with h5py.File(path, "r") as f:
        offsets = f["node_offsets"][:]
        dp_ids = f["dp_ids"][:]
        scalars = f["scalars"][:]
        total = f["node_data"].shape[0]
 
        if offsets[0] != 0 or offsets[-1] != total:
            fail(f"node_offsets spans [{offsets[0]}, {offsets[-1]}] but node_data has {total} rows")
        if len(offsets) != len(dp_ids) + 1:
            fail(f"node_offsets has {len(offsets)} entries for {len(dp_ids)} cases")
        if np.any(np.diff(offsets) <= 0):
            fail("node_offsets is not strictly increasing")
        if f["targets"].shape[0] != total or f["cell_ids"].shape[0] != total:
            fail("targets / cell_ids length disagrees with node_data")
 
        for i, dp_id in enumerate(dp_ids.tolist()):
            if not np.array_equal(scalars[i], param_table[dp_id]):
                fail(
                    f"/scalars row {i} does not match the table row for dp {dp_id}: "
                    f"{scalars[i].tolist()} vs {param_table[dp_id].tolist()}"
                )
 
 
def main() -> None:
    t0 = time.time()
    print(f"[pack] output -> {OUT_PATH}")
 
    param_table = load_param_table()
    case_paths = discover_cases()
    print(f"[pack] {len(case_paths)} CSV file(s), {len(param_table)} parameter rows")
    for dp_id in sorted(case_paths)[:5]:
        print(f"[pack]   {case_paths[dp_id].name}  ->  dp {dp_id}")
 
    # the join, both directions, before anything is written
    missing_params = sorted(set(case_paths) - set(param_table))
    missing_csvs = sorted(set(param_table) - set(case_paths))
    if missing_params or missing_csvs:
        fail(
            f"join incomplete -- {len(missing_params)} CSV(s) with no parameter row "
            f"{missing_params[:20]}, {len(missing_csvs)} parameter row(s) with no CSV "
            f"{missing_csvs[:20]}"
        )
    print("[pack] join OK")
 
    selected = sorted(case_paths)
    if LIMIT is not None:
        selected = selected[:LIMIT]
        print(f"[pack] LIMIT={LIMIT}: processing {len(selected)} of {len(case_paths)} cases")
 
    header = read_header(case_paths[selected[0]])
    if header != CSV_HEADER:
        fail(f"unexpected header:\n  expected {CSV_HEADER}\n  found    {header}")
 
    offsets = [0]
    dp_ids: list[int] = []
    scalars: list[np.ndarray] = []
    node_min = np.full(3, np.inf)
    node_max = np.full(3, -np.inf)
    target_min = np.full(2, np.inf)
    target_max = np.full(2, -np.inf)
 
    OUT_PATH.parent.mkdir(parents = True, exist_ok = True)
    with h5py.File(OUT_PATH, "w") as f:
        node_ds = f.create_dataset("node_data", shape = (0, 3), maxshape = (None, 3),
                                   dtype = np.float32, chunks = True)
        target_ds = f.create_dataset("targets", shape = (0, 2), maxshape = (None, 2),
                                     dtype = np.float32, chunks = True)
        cell_ds = f.create_dataset("cell_ids", shape = (0,), maxshape = (None,),
                                   dtype = np.int64, chunks = True)
 
        for k, dp_id in enumerate(selected):
            path = case_paths[dp_id]
            if read_header(path) != CSV_HEADER:
                fail(f"{path.name} has a different header from the first file")
 
            node, targets, cell_ids = read_case(path)
            n = node.shape[0]
            if n < MIN_ROWS_PER_CASE:
                fail(f"dp {dp_id}: only {n} rows -- truncated export?")
 
            start = offsets[-1]
            for dset, block in ((node_ds, node), (target_ds, targets), (cell_ds, cell_ids)):
                dset.resize(start + n, axis = 0)
                dset[start:] = block
 
            offsets.append(start + n)
            dp_ids.append(dp_id)
            scalars.append(param_table[dp_id])
            node_min = np.minimum(node_min, node.min(axis = 0))
            node_max = np.maximum(node_max, node.max(axis = 0))
            target_min = np.minimum(target_min, targets.min(axis = 0))
            target_max = np.maximum(target_max, targets.max(axis = 0))
 
            if k == 0:
                per_case = (node.nbytes + targets.nbytes + cell_ids.nbytes) / 1e6
                print(f"[pack] first case: {n:,} rows, {per_case:.1f} MB -- "
                      f"projected ~{per_case * len(selected) / 1000:.1f} GB")
            if k == 0 or (k + 1) % PROGRESS_EVERY == 0 or k + 1 == len(selected):
                rate = (k + 1) / max(time.time() - t0, 1e-9)
                print(f"[pack] {k + 1}/{len(selected)}  dp {dp_id}  {n:,} rows  "
                      f"total {offsets[-1]:,}  {rate:.2f} case/s")
 
        str_dtype = h5py.string_dtype(encoding = "utf-8")
        f.create_dataset("node_offsets", data = np.asarray(offsets, dtype = np.int64))
        f.create_dataset("dp_ids", data = np.asarray(dp_ids, dtype = np.int64))
        f.create_dataset("scalars", data = np.stack(scalars).astype(np.float32))
        for name, values in (("node_columns", NODE_COLUMNS),
                             ("target_columns", TARGET_COLUMNS),
                             ("scalar_columns", SCALAR_COLUMNS)):
            f.create_dataset(name, data = np.array(values, dtype = object), dtype = str_dtype)
        f.attrs["cropped"] = False
        f.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
 
    verify_written_file(OUT_PATH, param_table)
 
    counts = np.diff(offsets)
    print("\n" + "=" * 70)
    print(f"cases        : {len(dp_ids)}")
    print(f"total nodes  : {offsets[-1]:,}")
    print(f"rows per case: min {counts.min():,}  median {np.median(counts):,.0f}  max {counts.max():,}")
    print("node_data")
    for j, name in enumerate(NODE_COLUMNS):
        print(f"  {name:<20} [{node_min[j]:>14.6g}, {node_max[j]:>14.6g}]")
    print("targets")
    for j, name in enumerate(TARGET_COLUMNS):
        print(f"  {name:<20} [{target_min[j]:>14.6g}, {target_max[j]:>14.6g}]")
    print("scalars")
    arr = np.stack(scalars)
    for j, name in enumerate(SCALAR_COLUMNS):
        print(f"  {name:<20} [{arr[:, j].min():>14.6g}, {arr[:, j].max():>14.6g}]")
    print(f"dp IDs       : {min(dp_ids)} .. {max(dp_ids)}")
    print("=" * 70)
    print(f"[pack] done in {(time.time() - t0) / 60:.1f} min -> {OUT_PATH} "
          f"({OUT_PATH.stat().st_size / 1e9:.2f} GB)")
 
 
if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"\nFAILED: {exc}", file = sys.stderr)
        sys.exit(1)
                   
