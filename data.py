from __future__ import annotations

import hashlib
import json

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

MIN_NODES_AFTER_CROP = 16

# configuration

@dataclass(frozen = True)
class CropBox:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x >= self.x_min) & (x <= self.x_max) & (y >= self.y_min) & (y <= self.y_max)

@dataclass
class DataConfig:
    h5_path: str

    #column names
    node_columns: Sequence[str] = ("x", "y", "cell_volume")
    target_columns: Sequence[str] = ("static_pressure", "velocity_magnitude")
    scalar_columns: Sequence[str] = ("m", "p", "t", "AoA", "V_in")
    x_column: str = "x"
    y_column: str = "y"

    crop: Optional[CropBox] = None
    knn_k: int = 4
    cache_dir: str = "cache"

    split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_seed: int = 0

    def __post_init__(self) -> None:
        if abs(sum(self.split_fractions) - 1.0) > 1e-9:
            raise ValueError(f"split fractions must be sum to 1, got {self.split_fractions}")
        if self.knn_k < 1:
            raise ValueError("knn_k must be >=1")
        for name in (self.x_column, self.y_column):
            if name not in self.node_columns:
                raise ValueError(
                    f"Coordinate column {name!r} must also be one of node_columns"
                    f"{tuple(self.node_columns)}"
                )

    def cache_key(self, store_fingerprint: str) -> str:
        "Identifies a cached KNN build. Changing the crop or k changes the key"
        payload = {
            "store": store_fingerprint,
            "node_columns": list(self.node_columns),
            "x_column": self.x_column,
            "y_column": self.y_column,
            "crop": asdict(self.crop) if self.crop is not None else None,
            "knn_k": self.knn_k,
        }
        blob = json.dumps(payload, sort_keys = True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

# scaling

@dataclass
class MinMaxScaler:
    data_min: Optional[np.ndarray] = None
    data_max: Optional[np.ndarray] = None

    def partial_fit(self, batch: np.ndarray) -> "MinMaxScaler":
        batch = np.asarray(batch, dtype = np.float64)
        if batch.ndim != 2:
            raise ValueError (f"expected 2D array, got shape {batch.shape}")
        bmin, bmax = batch.min(axis = 0), batch.max(axis = 0)
        self.data_min = bmin if self.data_min is None else np.minimum(self.data_min, bmin)
        self.data_max = bmax if self.data_max is None else np.maximum(self.data_max, bmax)
        return self

    @property
    def _span(self) -> np.ndarray:
        span = self.data_max - self.data_min
        # map to 0 rather than / 0
        return np.where(span == 0.0, 1.0, span)

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype = np.float64) - self.data_min) / self._span
        return z.astype(np.float32)

    def inverse_transform(self, z):
        if isinstance(z, torch.Tensor):
            dmin = torch.as_tensor(self.data_min, dtype = z.dtype, device = z.device)
            span = torch.as_tensor(self._span, dtype = z.dtype, device = z.device)
            return z * span + dmin
        return (np.asarray(z, dtype = np.float64) * self._span + self.data_min).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "data_min": np.asarray(self.data_min).tolist(),
            "data_max": np.asarray(self.data_max).tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MinMaxScaler":
        return cls(
            data_min = np.asarray(d["data_min"], dtype = np.float64),
            data_max = np.asarray(d["data_max"], dtype = np.float64),
        )


@dataclass
class ScalerBundle:
    node: MinMaxScaler
    target: MinMaxScaler
    scalar: MinMaxScaler
    node_columns: list[str] = field(default_factory = list)
    target_columns: list[str] = field(default_factory = list)
    scalar_columns: list[str] = field(default_factory = list)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text(
            json.dumps(
                {
                "node": self.node.to_dict(),
                "target": self.target.to_dict(),
                "scalar": self.scalar.to_dict(),
                "node_columns": self.node_columns,
                "target_columns": self.target_columns,
                "scalar_columns": self.scalar_columns,
                },
                indent = 2,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "ScalerBundle":
        d = json.loads(Path(path).read_text())
        return cls(
            node = MinMaxScaler.from_dict(d["node"]),
            target = MinMaxScaler.from_dict(d["target"]),
            scalar = MinMaxScaler.from_dict(d["scalar"]),
            node_columns = d["node_columns"],
            target_columns = d["target_columns"],
            scalar_columns = d["scalar_columns"],
        )

#HDF5 access

def _decode_names(dataset) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in dataset[:]]

def _select_by_name(available: Sequence[str], requested: Sequence[str], what: str) -> list[int]:
    lookup: dict[str, int] = {}
    for i, name in enumerate(available):
        if name in lookup:
            raise ValueError(f"duplicate {what} column name {name!r} in the HDF5 file")
        lookup[name] = i

    missing = [n for n in requested if n not in lookup]
    if missing:
        raise KeyError(
            f"{what} columns {missing} not found in the file. Available: {list(available)}"
        )
    return [lookup[n] for n in requested]

class CaseStore:
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self.path = Path(cfg.h5_path)
        if not self.path.exists():
            raise FileNotFoundError(f"HDF5 dataset not found: {self.path}")

        with h5py.File(self.path, "r") as f:
            for required in (
                "node_data", "node_offsets", "targets", "scalars", "dp_ids", "cell_ids", "node_columns", "target_columns", "scalar_columns",
            ):
                if required not in f:
                    raise KeyError (f"dataset /{required} missing from {self.path}")

            self.available_node_columns = _decode_names(f["node_columns"])
            self.available_target_columns = _decode_names(f["target_columns"])
            self.available_scalar_columns = _decode_names(f["scalar_columns"])
            self.node_offsets = np.asarray(f["node_offsets"][:], dtype=np.int64)
            self.dp_ids = np.asarray(f["dp_ids"][:], dtype=np.int64)
            self.scalars_raw = np.asarray(f["scalars"][:], dtype=np.float64)
            self.total_nodes = int(f["node_data"].shape[0])
            n_node_cols = int(f["node_data"].shape[1])

        if n_node_cols != len(self.available_node_columns):
            raise ValueError(
                f"/node_data has {n_node_cols} columns but /node_columns lists "
                f"{len(self.available_node_columns)} names"
            )
        if self.node_offsets[0] != 0 or self.node_offsets[-1] != self.total_nodes:
            raise ValueError("/node_offsets does not span /node_data exactly")

        self.node_idx = _select_by_name(self.available_node_columns, cfg.node_columns, "node")
        self.target_idx = _select_by_name(
            self.available_target_columns, cfg.target_columns, "target"
        )
        self.scalar_idx = _select_by_name(
            self.available_scalar_columns, cfg.scalar_columns, "scalar"
        )

        # Position of x and y inside the selected node feature matrix
        self.x_pos = list(cfg.node_columns).index(cfg.x_column)
        self.y_pos = list(cfg.node_columns).index(cfg.y_column)

        self.n_cases = len(self.node_offsets) - 1
        if self.scalars_raw.shape[0] != self.n_cases:
            raise ValueError(
                f"/scalars has {self.scalars_raw.shape[0]} rows but there are "
                f"{self.n_cases} cases"
            )
        self._file: Optional[h5py.File] = None

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def fingerprint(self) -> str:
        stat = self.path.stat()
        return f"{self.path.name}: {stat.st_size}: {int(stat.st_mtime)}: {self.n_cases}"

    def raw_case(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        "Return (node_features, targets, scalars, dp_id) for case ``i``, uncropped."
        a, b = int(self.node_offsets[i]), int(self.node_offsets[i + 1])
        f = self.file
        nodes = np.asarray(f["node_data"][a:b, :], dtype=np.float64)[:, self.node_idx]
        targets = np.asarray(f["targets"][a:b, :], dtype=np.float64)[:, self.target_idx]
        scalars = self.scalars_raw[i, self.scalar_idx]
        return nodes, targets, scalars, int(self.dp_ids[i])

    def cropped_case(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        nodes, targets, scalars, dp_id = self.raw_case(i)
        crop = self.cfg.crop
        if crop is not None:
            keep = crop.mask(nodes[:, self.x_pos], nodes[:, self.y_pos])
            n_kept = int(keep.sum())
            if n_kept < MIN_NODES_AFTER_CROP:
                raise ValueError(
                    f"case {i} (dp_id={dp_id}) has only {n_kept} nodes inside the crop "
                    f"box {crop}; minimum is {MIN_NODES_AFTER_CROP}"
                )
            nodes, targets = nodes[keep], targets[keep]
        return nodes, targets, scalars, dp_id

# graph construction and caching

def knn_edge_index(xy: np.ndarray, k: int) -> np.ndarray:
    "Symmetric KNN connectivity on 2D points, returned as a (2, E) COO array"

    n = xy.shape[0]
    k_eff = min(k, n - 1)
    if k_eff < 1:
        return np.zeros((2,0), dtype = np.int64)

    tree = cKDTree(xy)
    _, idx = tree.query(xy, k = k_eff + 1)
    idx = np.atleast_2d(idx)
    src = np.repeat(np.arange(n, dtype = np.int64), k_eff)
    dst = idx[:, 1:].reshape(-1).astype(np.int64)

    src, dst = np.concatenate([src, dst]), np.concatenate([dst, src])
    keep = src != dst
    pairs = np.unique(np.stack([src[keep], dst[keep]], axis = 1), axis = 0)
    return pairs.T.astype(np.int64)

class GraphCache:
    "On-disk cache of the KNN connectivity, keyed by the crop + KNN configuration."
    def __init__(self, cfg: DataConfig, store: CaseStore) -> None:
        self.key = cfg.cache_key(store.fingerprint)
        self.path = Path(cfg.cache_dir) / f"knn_{self.key}.npz"
        self.cfg = cfg
        self.store = store
        self._edges: Optional[list[np.ndarray]] = None

    def build_or_load(self, verbose: bool = True) -> None:
        if self.path.exists():
            with np.load(self.path) as z:
                flat, offsets = z["edges"], z["edge_offsets"]
            self._edges = [
                flat[:, int(offsets[i]): int(offsets[i + 1])] for i in range(len(offsets) - 1)
            ]
            if verbose:
                print(f"[data] loaded cached KNN graphs from {self.path}")
            return

        if verbose:
            print(f"[data] building KNN graphs (k={self.cfg.knn_k}) -> {self.path}")
        edges = []
        for i in range(self.store.n_cases):
            nodes, _, _, _ = self.store.cropped_case(i)
            xy = nodes[:, [self.store.x_pos, self.store.y_pos]]
            edges.append(knn_edge_index(xy, self.cfg.knn_k))

        offsets = np.cumsum([0] + [e.shape[1] for e in edges]).astype(np.int64)
        flat = np.concatenate(edges, axis=1) if edges else np.zeros((2, 0), dtype=np.int64)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, edges=flat, edge_offsets=offsets)
        self._edges = edges

    def __getitem__(self, i: int) -> np.ndarray:
        if self._edges is None:
            self.build_or_load()
        return self._edges[i]

# Dataset

class GraphDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            store: CaseStore,
            cache: GraphCache,
            indices: Sequence[int],
            scalers: ScalerBundle,
    ) -> None:
        self.store = store
        self.cache = cache
        self.indices = list(indices)
        self.scalers = scalers

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, j:int) -> Data:
        i = self.indices[j]
        nodes, targets, scalars, dp_id = self.store.cropped_case(i)
        edge_index = self.cache[i]
        if edge_index.size and edge_index.max() >= nodes.shape[0]:
            raise RuntimeError(
                f"cached graph for case {i} references node {int(edge_index.max())} but the cropped case has {nodes.shape[0]} nodes -- the cache is stale for this crop configuration"
            )

        data = Data(
            x = torch.from_numpy(self.scalers.node.transform(nodes)),
            edge_index = torch.from_numpy(np.ascontiguousarray(edge_index)),
            y = torch.from_numpy(self.scalers.target.transform(targets)),
            num_nodes = nodes.shape[0]
        )
        data.scalars = torch.from_numpy(self.scalers.scalar.transform(scalars[None, :]))
        data.dp_id = torch.tensor([dp_id], dtype=torch.long)
        data.case_index = torch.tensor([i], dtype=torch.long)
        return data

def _split_indices(
    n: int, fractions: tuple[float, float, float], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = min(int(round(fractions[0] * n)), n)
    n_val = min(int(round(fractions[1] * n)), n - n_train)
    return perm[:n_train], perm[n_train: n_train + n_val], perm[n_train + n_val:]

def fit_scalers(store: CaseStore, train_indices: Sequence[int], cfg: DataConfig) -> ScalerBundle:
    "Fit min-max scalers on the training split only, after cropping"

    node, target, scalar = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()

    scalar_rows = []
    for i in train_indices:
        nodes, targets, scalars, _ = store.cropped_case(int(i))
        node.partial_fit(nodes)
        target.partial_fit(targets)
        scalar_rows.append(scalars)
    scalar.partial_fit(np.stack(scalar_rows))

    return ScalerBundle(
        node=node,
        target=target,
        scalar=scalar,
        node_columns=list(cfg.node_columns),
        target_columns=list(cfg.target_columns),
        scalar_columns=list(cfg.scalar_columns),
    )

def build_splits(
    cfg: DataConfig, verbose: bool = True
) -> tuple[GraphDataset, GraphDataset, GraphDataset, ScalerBundle, CaseStore]:
    "Load, crop, build/load the KNN cache, fit scalers on train, return the splits."
    store = CaseStore(cfg)
    cache = GraphCache(cfg, store)
    cache.build_or_load(verbose=verbose)

    train_idx, val_idx, test_idx = _split_indices(
        store.n_cases, cfg.split_fractions, cfg.split_seed
    )
    if len(train_idx) == 0:
        raise ValueError("training split is empty")
    scalers = fit_scalers(store, train_idx, cfg)

    if verbose:
        print(
            f"[data] {store.n_cases} cases -> "
            f"{len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test"
        )
        if cfg.crop is not None:
            print(f"[data] crop box: {cfg.crop}")
        print(
            "[data] target min/max (fitted on train, after crop): "
            + ", ".join(
                f"{n}=[{lo:.4g}, {hi:.4g}]"
                for n, lo, hi in zip(
                    cfg.target_columns, scalers.target.data_min, scalers.target.data_max
                )
            )
        )

    return (
        GraphDataset(store, cache, train_idx, scalers),
        GraphDataset(store, cache, val_idx, scalers),
        GraphDataset(store, cache, test_idx, scalers),
        scalers,
        store,
    )
