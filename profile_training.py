from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from data import CropBox, DataConfig, build_splits
from model import GCNSurrogate, GCNSurrogateConfig, count_parameters

# settings 

H5_PATH = r"C:\Users\26664984\Documents\Masters\hdf5_training_data\iteration2\hydrofoil_rev2.h5"
CROP =  (-0.2, 1.50, 0.80, 1.4)
KNN_K = 4
CACHE_DIR = "cache"

N_CASES = 50            # a subset is enough to get a per-case rate
N_WARMUP = 5            # CUDA kernels are compiled on first use; discard those
WORKER_COUNTS = [0, 2, 4]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sync() -> None:
    "CUDA calls are asynchronous, so timing without this measures queueing, not work"
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    print(f"[profile] device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"[profile] {torch.cuda.get_device_name(0)}")

    data_cfg = DataConfig(
        h5_path = H5_PATH,
        crop = CropBox(*CROP),
        knn_k = KNN_K,
        cache_dir = CACHE_DIR,
    )
    train_ds, _, _, scalers, store = build_splits(data_cfg, verbose = True)

    model_cfg = GCNSurrogateConfig(
        n_scalar_features = len(data_cfg.scalar_columns),
        n_node_features = len(data_cfg.node_columns),
        n_outputs = len(data_cfg.target_columns),
    )
    model = GCNSurrogate(model_cfg).to(DEVICE)
    loss_fn = lambda p, t: 0.5 * nn.functional.mse_loss(p, t)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    print(f"[profile] {count_parameters(model):,} parameters")

    # ---- 1. the floor: one graph, already on the GPU, no loader -------------------
    sample = train_ds[0].to(DEVICE)
    n_nodes = sample.num_nodes
    n_edges = sample.edge_index.shape[1]
    print(f"\n[profile] sample case: {n_nodes:,} nodes, {n_edges:,} edges")

    for _ in range(N_WARMUP):
        optimizer.zero_grad(set_to_none = True)
        loss = loss_fn(model(sample.x, sample.edge_index, sample.scalars, None), sample.y)
        loss.backward()
        optimizer.step()
    sync()

    t0 = time.perf_counter()
    for _ in range(N_CASES):
        optimizer.zero_grad(set_to_none = True)
        loss = loss_fn(model(sample.x, sample.edge_index, sample.scalars, None), sample.y)
        loss.backward()
        optimizer.step()
    sync()
    gpu_only = (time.perf_counter() - t0) / N_CASES

    print(f"\n{'=' * 72}")
    print(f"1. forward + backward only, graph resident on device")
    print(f"   {gpu_only * 1000:8.1f} ms per case")
    print(f"   {gpu_only * len(train_ds) / 60:8.1f} min per epoch at this rate")
    print(f"   {gpu_only * len(train_ds) * 1000 / 3600:8.1f} hours for 1000 epochs")

    # ---- 2 and 3. through the DataLoader, with and without workers ----------------
    print(f"\n{'=' * 72}")
    print("2. through the DataLoader (loading + transfer + model)")
    print(f"{'workers':>9} | {'ms/case':>9} | {'min/epoch':>10} | {'h/1000ep':>9} | overhead")
    print("-" * 72)

    for workers in WORKER_COUNTS:
        loader = DataLoader(train_ds, batch_size = 1, shuffle = True,
                            num_workers = workers,
                            persistent_workers = workers > 0)
        it = iter(loader)
        for _ in range(N_WARMUP):          # worker startup is a one-off cost
            next(it)
        sync()

        t0 = time.perf_counter()
        for i, batch in enumerate(it):
            if i >= N_CASES:
                break
            batch = batch.to(DEVICE)
            optimizer.zero_grad(set_to_none = True)
            loss = loss_fn(
                model(batch.x, batch.edge_index, batch.scalars, batch.batch), batch.y
            )
            loss.backward()
            optimizer.step()
        sync()
        full = (time.perf_counter() - t0) / N_CASES

        print(f"{workers:>9} | {full * 1000:9.1f} | "
              f"{full * len(train_ds) / 60:10.1f} | "
              f"{full * len(train_ds) * 1000 / 3600:9.1f} | "
              f"{full / gpu_only:.1f}x")

    print("=" * 72)
    print("If the overhead column is near 1x, the GPU is the limit and caching the")
    print("dataset would not help. If it is 3x or more, most of the time is going into")
    print("reading and converting data, and holding the cases in memory would remove it.")

    if DEVICE == "cuda":
        print(f"\n[profile] peak GPU memory: "
              f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB of "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    store.close()


if __name__ == "__main__":
    main()