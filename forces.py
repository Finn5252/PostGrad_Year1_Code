# libraries

from __future__ import annotations

import json

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.loader import DataLoader

from data import build_splits, targets_to_physical
from evaluate import DEVICE, load_run
from geometry import CHORD, MATCH_RADIUS, QUERY_OFFSET, foil_outline, panels
from settings import RHO, RUN_DIR

# settings

SPLIT = "test" # "train", "val" or "test"


def integrate(
    xy: np.ndarray,
    pressure: np.ndarray,
    outline: np.ndarray
) -> tuple[float, float, float]:
    "Return (lift, pressure drag, fraction of panels matched)"
    mid, normal, ds = panels(outline)
    tree = cKDTree(xy)
    # the pressure on a panel is taken from the cell centre nearest to a point just outside it
    probe = mid + QUERY_OFFSET * normal
    dist, idx = tree.query(probe)

    ok = dist < MATCH_RADIUS
    if not ok.any():
        return float("nan"), float("nan"), 0.0

    p = pressure[idx[ok]]
    # pressure pushes against the outward normal, hence the minus sign
    force = -(p[:, None] * normal[ok] * ds[ok, None]).sum(axis = 0)

    return float(force[1]), float(force[0]), float(ok.mean())


def force_coefficient(force: float, v_in: float) -> float:
    "Lift or drag over the dynamic pressure times the chord"
    return force / (0.5 * RHO * v_in**2 * CHORD)


@torch.no_grad()
def main() -> None:
    model, data_cfg = load_run(tag = "forces")
    train_ds, val_ds, test_ds, scalers, store = build_splits(data_cfg, verbose = False)
    dataset = {"train": train_ds, "val": val_ds, "test": test_ds}[SPLIT]
    loader = DataLoader(dataset, batch_size = 1)

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
        p_pred = targets_to_physical(pred, batch, scalers).cpu().numpy()[:, i_pressure]
        p_true = targets_to_physical(batch.y, batch, scalers).cpu().numpy()[:, i_pressure]
        s = scalers.scalar.inverse_transform(batch.scalars.cpu().numpy())[0]
        v_in = float(s[i_v])

        outline = foil_outline(float(s[i_m]), float(s[i_p]), float(s[i_t]),
                               float(s[i_aoa]))
        # the same integration applied to both, so its own error cancels in the comparison
        lift_p, drag_p, matched = integrate(xy, p_pred, outline)
        lift_t, drag_t, _ = integrate(xy, p_true, outline)

        rows.append({
            "dp_id": int(batch.dp_id[0].item()),
            "V_in": v_in,
            "Cl_cfd": force_coefficient(lift_t, v_in),
            "Cl_pred": force_coefficient(lift_p, v_in),
            "Cdp_cfd": force_coefficient(drag_t, v_in),
            "Cdp_pred": force_coefficient(drag_p, v_in),
            "panels_matched": matched,
        })

    cl_t = np.array([r["Cl_cfd"] for r in rows])
    cl_p = np.array([r["Cl_pred"] for r in rows])
    cdp_t = np.array([r["Cdp_cfd"] for r in rows])
    cdp_p = np.array([r["Cdp_pred"] for r in rows])
    matched = np.array([r["panels_matched"] for r in rows])

    def summary(true, pred, label):
        err = pred - true
        rel = 100.0 * np.abs(err) / np.maximum(np.abs(true), 1e-12) # np.maximum guards a CFD value of 0
        print(f"  {label}")
        print(f"    CFD range        {true.min():>9.4f} to {true.max():>9.4f}")
        print(f"    mean |error|     {np.abs(err).mean():>9.4f}  "
              f"({rel.mean():.2f}% of the CFD value)")
        print(f"    worst case       {np.abs(err).max():>9.4f}  "
              f"(dp {rows[int(np.abs(err).argmax())]['dp_id']})")
        print(f"    bias             {err.mean():>+9.4f}")

    print(f"Integrated forces, {SPLIT} split ({len(rows)} cases)")
    print(f"  panels matched to a cell: {matched.mean():.1%} "
          f"(worst case {matched.min():.1%})")
    summary(cl_t, cl_p, "lift coefficient Cl")
    summary(cdp_t, cdp_p, "pressure drag coefficient Cdp")
    out = RUN_DIR / f"forces_{SPLIT}.json"
    out.write_text(json.dumps(rows, indent = 2))
    print(f"[forces] per-case values -> {out.name}")
    store.close()


if __name__ == "__main__":
    main()
