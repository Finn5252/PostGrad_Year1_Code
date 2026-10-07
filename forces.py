# libraries

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.loader import DataLoader

from data import CropBox, DataConfig, build_splits
from model import GCNSurrogate, GCNSurrogateConfig

# settings

RUN_DIR = Path(r"C:\Users\26664984\Documents\Masters\Model_training\iter1")
H5_PATH = r"C:\Users\26664984\Documents\Masters\hdf5_training_data\hydrofoil.h5"

SPLIT = "test" # use test distribution 

RHO = 1025.0 # working fluid density
CHORD = 1.0 # reference length for the coefficients

LEADING_EDGE_X = 0.0 # position of the leading edge in the CFD domain
LEADING_EDGE_Y = 1.0

N_PANELS = 200# outline points per surface
QUERY_OFFSET = 1e-4 # how far outside the surface to look for the adjacent cell.



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# geometry

def naca4_profile(m: float, p: float, t: float, n: int = N_PANELS) -> np.ndarray:
    "Closed NACA 4-digit outline, upper surface then lower, as (2n, 2) coordinates"
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))

    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                    + 0.2843 * x**3 - 0.1015 * x**4)

    if m == 0.0 or p == 0.0:
        yc = np.zeros_like(x) # symmetric section: no camber line
        dyc = np.zeros_like(x)
    else:
        fore = x <= p
        yc = np.where(fore,
                      m / p**2 * (2 * p * x - x**2),
                      m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
        dyc = np.where(fore,
                       2 * m / p**2 * (p - x),
                       2 * m / (1 - p)**2 * (p - x))

    # thickness added normal to the camber line, not vertically
    theta = np.arctan(dyc)
    xu, yu = x - yt * np.sin(theta), yc + yt * np.cos(theta)
    xl, yl = x + yt * np.sin(theta), yc - yt * np.cos(theta)

    # lower surface reversed, so the two join into one closed loop
    return np.concatenate([
        np.stack([xu, yu], axis = 1),
        np.stack([xl[::-1], yl[::-1]], axis = 1),
    ])


def foil_outline(m: float, p: float, t: float, aoa_deg: float) -> np.ndarray:
    "The outline rotated about the leading edge and placed in the CFD domain"
    xy = naca4_profile(m, p, t)
    a = np.radians(aoa_deg)
    c, s = np.cos(-a), np.sin(-a) # negative: nose-up for a positive angle
    rot = np.stack([xy[:, 0] * c - xy[:, 1] * s,
                    xy[:, 0] * s + xy[:, 1] * c], axis = 1)
    return rot + np.array([LEADING_EDGE_X, LEADING_EDGE_Y])


def panels(outline: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    "Segment midpoints, outward unit normals and lengths."
    a = outline
    b = np.roll(outline, -1, axis = 0)  
    d = b - a                          
    length = np.linalg.norm(d, axis = 1)

    keep = length > 1e-12 # drop zero-length segments, e.g. a sharp TE
    a, b, d, length = a[keep], b[keep], d[keep], length[keep]

    signed_area = 0.5 * np.sum(a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1])
    if signed_area > 0: # anticlockwise
        normal = np.stack([d[:, 1], -d[:, 0]], axis = 1)
    else: # clockwise
        normal = np.stack([-d[:, 1], d[:, 0]], axis = 1)
    normal /= length[:, None]

    return 0.5 * (a + b), normal, length


# forces

def integrate(xy: np.ndarray, pressure: np.ndarray, outline: np.ndarray) -> tuple[float, float]:
    "Return (lift per unit depth, fraction of panels matched)."
    mid, normal, ds = panels(outline)
    tree = cKDTree(xy)
    probe = mid + QUERY_OFFSET * normal
    dist, idx = tree.query(probe)

    ok = dist < 50 * QUERY_OFFSET
    if not ok.any():
        return float("nan"), 0.0

    p = pressure[idx[ok]]
    force = -(p[:, None] * normal[ok] * ds[ok, None]).sum(axis = 0)
    return float(force[1]), float(ok.mean())


def lift_coefficient(lift: float, v_in: float) -> float:
    return lift / (0.5 * RHO * v_in**2 * CHORD)


def load_run() -> tuple[GCNSurrogate, DataConfig]:
    saved = json.loads((RUN_DIR / "config.json").read_text())
    d = saved["data"]
    crop = d.get("crop")
    data_cfg = DataConfig(
        h5_path = H5_PATH,
        node_columns = tuple(d["node_columns"]),
        target_columns = tuple(d["target_columns"]),
        scalar_columns = tuple(d["scalar_columns"]),
        crop = CropBox(**crop) if crop else None,
        knn_k = d["knn_k"],
        cache_dir = d["cache_dir"],
        split_fractions = tuple(d["split_fractions"]),
        split_seed = d["split_seed"],
    )
    model = GCNSurrogate(GCNSurrogateConfig(**saved["model"])).to(DEVICE)
    ckpt = torch.load(RUN_DIR / "best.pt", map_location = DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[forces] loaded epoch {ckpt['epoch']} on {DEVICE}")
    return model, data_cfg


@torch.no_grad()
def main() -> None:
    model, data_cfg = load_run()
    train_ds, val_ds, test_ds, scalers, store = build_splits(data_cfg, verbose = False)
    dataset = {"train": train_ds, "val": val_ds, "test": test_ds}[SPLIT]
    loader = DataLoader(dataset, batch_size = 1)
    print(f"[forces] {SPLIT} split: {len(dataset)} cases")

    names = list(scalers.scalar_columns)
    i_m, i_p, i_t = names.index("m"), names.index("p"), names.index("t")
    i_aoa, i_v = names.index("AoA"), names.index("V_in")
    i_pressure = list(scalers.target_columns).index("static_pressure")

    rows = []
    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch.x, batch.edge_index, batch.scalars, batch.batch)

        # everything back to physical units
        xy = scalers.node.inverse_transform(batch.x.cpu().numpy())[:, :2]
        p_pred = scalers.target.inverse_transform(pred).cpu().numpy()[:, i_pressure]
        p_true = scalers.target.inverse_transform(batch.y).cpu().numpy()[:, i_pressure]
        s = scalers.scalar.inverse_transform(batch.scalars.cpu().numpy())[0]

        outline = foil_outline(float(s[i_m]), float(s[i_p]), float(s[i_t]),
                               float(s[i_aoa]))
        # The same integration applied to both
        lift_p, matched = integrate(xy, p_pred, outline)
        lift_t, _ = integrate(xy, p_true, outline)

        rows.append({
            "dp_id": int(batch.dp_id[0].item()),
            "V_in": float(s[i_v]),
            "Cl_cfd": lift_coefficient(lift_t, float(s[i_v])),
            "Cl_pred": lift_coefficient(lift_p, float(s[i_v])),
            "panels_matched": matched,
        })

    cl_t = np.array([r["Cl_cfd"] for r in rows])
    cl_p = np.array([r["Cl_pred"] for r in rows])
    matched = np.array([r["panels_matched"] for r in rows])

    def summary(true, pred, label):
        err = pred - true
        rel = 100.0 * np.abs(err) / np.maximum(np.abs(true), 1e-12)
        print(f"  {label}")
        print(f"    CFD range        {true.min():>9.4f} to {true.max():>9.4f}")
        print(f"    mean |error|     {np.abs(err).mean():>9.4f}  "
              f"({rel.mean():.2f}% of the CFD value)")
        print(f"    worst case       {np.abs(err).max():>9.4f}  "
              f"(dp {rows[int(np.abs(err).argmax())]['dp_id']})")
        print(f"    bias             {err.mean():>+9.4f}")

    print("\n" + "=" * 72)
    print(f"Integrated lift, {SPLIT} split ({len(rows)} cases)")
    print("=" * 72)
    print(f"  panels matched to a cell: {matched.mean():.1%} "
          f"(worst case {matched.min():.1%})")
    summary(cl_t, cl_p, "lift coefficient Cl")
    print("=" * 72)
    print("Both columns are integrated the same way, so this isolates the surrogate's")
    print("error. Validate the integration itself by comparing Cl_cfd against the lift")
    print("reported by Fluent (P19 in the design point table) before trusting Cl_pred.")

    out = RUN_DIR / f"lift_{SPLIT}.json"
    out.write_text(json.dumps(rows, indent = 2))
    print(f"[forces] per-case values -> {out.name}")
    store.close()


if __name__ == "__main__":
    main()