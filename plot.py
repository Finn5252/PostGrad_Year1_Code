from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# settings

RUN_DIR = Path(r"C:\Users\26664984\Documents\Masters\Model_training\iter1")
FIELD_DIR = RUN_DIR / "fields"
FIG_DIR = RUN_DIR / "figures"

GRID_N = 400 # interpolation grid resolution along x
SMOOTH_SIGMA = 1.0  # gaussian smoothing of the interpolated grid
N_CONTOURS = 20
OVERLAY_EVERY = 2 #every Nth contour level in the overlay panel

ZERO_THRESHOLD = 1e-12  # near-zero floor for the relative L1/L2
ZERO_MODE = "absolute"  

MASK_RADIUS = 0.01      # grid points further than this from any cell centre are masked

DPI = 150

FIELD_LABELS = {
    "static_pressure": ("Pressure", "kPa", 1e-3),      # name, unit, scale factor
    "velocity_magnitude": ("Velocity magnitude", "m/s", 1.0),
}


def load_case(path: Path) -> dict:
    z = np.load(path, allow_pickle = True)
    return {
        "x": z["x"],
        "y": z["y"],
        "pred": z["pred"],
        "true": z["true"],
        "scalars": z["scalars"],
        "target_columns": [str(c) for c in z["target_columns"]],
        "scalar_columns": [str(c) for c in z["scalar_columns"]],
        "dp_id": int(path.stem.replace("dp", "")),
    }


def to_grid(x, y, values):
    "Interpolate scattered cell-centre values onto a regular grid, masking empty regions"
    xi = np.linspace(x.min(), x.max(), GRID_N)
    aspect = (y.max() - y.min()) / (x.max() - x.min())
    yi = np.linspace(y.min(), y.max(), max(int(GRID_N * aspect), 50))
    XI, YI = np.meshgrid(xi, yi)

    ZI = griddata((x, y), values, (XI, YI), method = "linear")

    # Mask grid points with no nearby data: the foil interior, and the corners where the
    # crop box does not quite reach.
    from scipy.spatial import cKDTree
    tree = cKDTree(np.stack([x, y], axis = 1))
    dist, _ = tree.query(np.stack([XI.ravel(), YI.ravel()], axis = 1))
    ZI = np.ma.masked_where(
        (dist.reshape(XI.shape) > MASK_RADIUS) | np.isnan(ZI), ZI
    )

    if SMOOTH_SIGMA > 0:
        from scipy.ndimage import gaussian_filter
        filled = ZI.filled(np.nan)
        valid = ~np.isnan(filled)
        smoothed = filled.copy()
        smoothed[~valid] = 0.0
        num = gaussian_filter(smoothed, SMOOTH_SIGMA)
        den = gaussian_filter(valid.astype(float), SMOOTH_SIGMA)
        with np.errstate(invalid = "ignore", divide = "ignore"):
            out = num / den
        ZI = np.ma.masked_where(~valid, out)

    return XI, YI, ZI


def case_title(case: dict) -> str:
    pairs = zip(case["scalar_columns"], case["scalars"])
    return f"dp {case['dp_id']}  (" + ", ".join(f"{n}={v:.3g}" for n, v in pairs) + ")"


def plot_contours(case: dict, field_index: int) -> None:
    "Predicted, actual, and the two overlaid"
    name = case["target_columns"][field_index]
    label, unit, scale = FIELD_LABELS.get(name, (name, "", 1.0))

    pred = case["pred"][:, field_index] * scale
    true = case["true"][:, field_index] * scale
    lo, hi = min(pred.min(), true.min()), max(pred.max(), true.max())
    levels = np.linspace(lo, hi, N_CONTOURS)

    XI, YI, ZP = to_grid(case["x"], case["y"], pred)
    _, _, ZT = to_grid(case["x"], case["y"], true)

    fig, ax = plt.subplots(3, 1, figsize = (9, 10), sharex = True, sharey = True)

    for a, Z, title in ((ax[0], ZP, "Predicted"), (ax[1], ZT, "Actual (CFD)")):
        cf = a.contourf(XI, YI, Z, levels = levels, cmap = "viridis", extend = "both")
        a.contour(XI, YI, Z, levels = levels, colors = "k", linewidths = 0.25)
        a.set_title(title)
        a.set_ylabel("y")
        a.set_aspect("equal")
        fig.colorbar(cf, ax = a, label = f"{label} [{unit}]" if unit else label)

    # overlay: the two sets of contour lines on the same axes
    sparse = levels[::OVERLAY_EVERY]
    ax[2].contour(XI, YI, ZT, levels = sparse, colors = "k", linewidths = 0.8)
    ax[2].contour(XI, YI, ZP, levels = sparse, colors = "r", linewidths = 0.8,
                  linestyles = "dashed")
    ax[2].set_title(f"Overlay -- black: CFD, red dashed: predicted "
                    f"(every {OVERLAY_EVERY}nd level)")
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("y")
    ax[2].set_aspect("equal")

    fig.suptitle(f"{label} -- {case_title(case)}", fontsize = 10)
    fig.tight_layout()
    out = FIG_DIR / f"dp{case['dp_id']}_{name}_contours.png"
    fig.savefig(out, dpi = DPI, bbox_inches = "tight")
    plt.close(fig)
    print(f"[plot] {out.name}")


def plot_error(case: dict, field_index: int) -> None:
    "Signed difference and relative error"
    name = case["target_columns"][field_index]
    label, unit, scale = FIELD_LABELS.get(name, (name, "", 1.0))

    pred = case["pred"][:, field_index] * scale
    true = case["true"][:, field_index] * scale
    diff = pred - true

    rms = float(np.sqrt((true**2).mean()))
    thr = ZERO_THRESHOLD if ZERO_MODE == "absolute" else ZERO_THRESHOLD * rms
    keep = np.abs(true) > thr
    ratio = np.abs(diff[keep]) / np.abs(true[keep])
    l1 = float(ratio.mean())
    l2 = float(np.sqrt((ratio**2).mean()))
    excluded = 1.0 - keep.mean()

    rel = np.full_like(diff, np.nan)
    rel[keep] = 100.0 * ratio

    fig, ax = plt.subplots(2, 1, figsize = (9, 7), sharex = True, sharey = True)

    span = np.abs(diff).max()
    XI, YI, ZD = to_grid(case["x"], case["y"], diff)
    cf = ax[0].contourf(XI, YI, ZD, levels = np.linspace(-span, span, 41),
                        cmap = "RdBu_r", extend = "both")
    ax[0].set_title(f"Predicted - actual   (max |error| = {span:.4g} {unit})")
    fig.colorbar(cf, ax = ax[0], label = f"difference [{unit}]" if unit else "difference")

    XI, YI, ZR = to_grid(case["x"], case["y"], rel)
    cf = ax[1].contourf(XI, YI, ZR, levels = np.linspace(0, min(np.nanmax(ZR), 50), 26),
                        cmap = "magma_r", extend = "max")
    ax[1].set_title(f"Relative error   L1 = {l1:.4f}   L2 = {l2:.4f}")
    fig.colorbar(cf, ax = ax[1], label = "relative error [%]")
    ax[1].text(0.01, -0.22, f"|phi| > {thr:g} ({ZERO_MODE}), "
               f"{excluded:.2%} of nodes excluded",
               transform = ax[1].transAxes, fontsize = 8, color = "0.35")

    for a in ax:
        a.set_ylabel("y")
        a.set_aspect("equal")
    ax[1].set_xlabel("x")

    fig.suptitle(f"{label} error -- {case_title(case)}", fontsize = 10)
    fig.tight_layout()
    out = FIG_DIR / f"dp{case['dp_id']}_{name}_error.png"
    fig.savefig(out, dpi = DPI, bbox_inches = "tight")
    plt.close(fig)
    print(f"[plot] {out.name}  L1={l1:.4f} L2={l2:.4f}")


def plot_parity(cases: list[dict]) -> None:
    "Predicted against actual for every node of every exported case, shaded by camber"
    n_fields = len(cases[0]["target_columns"])
    fig, ax = plt.subplots(1, n_fields, figsize = (6 * n_fields, 5.5))
    ax = np.atleast_1d(ax)

    camber_index = cases[0]["scalar_columns"].index("m")

    for k in range(n_fields):
        name = cases[0]["target_columns"][k]
        label, unit, scale = FIELD_LABELS.get(name, (name, "", 1.0))
        for case in cases:
            t = case["true"][:, k] * scale
            p = case["pred"][:, k] * scale
            m = np.full(t.shape, case["scalars"][camber_index])
            sc = ax[k].scatter(t, p, c = m, s = 0.3, alpha = 0.3,
                               cmap = "viridis", vmin = 0.0, vmax = 0.04,
                               rasterized = True)

        lo = min(ax[k].get_xlim()[0], ax[k].get_ylim()[0])
        hi = max(ax[k].get_xlim()[1], ax[k].get_ylim()[1])
        ax[k].plot([lo, hi], [lo, hi], "r--", linewidth = 1, label = "1:1")
        band = 0.1 * max(abs(lo), abs(hi))
        ax[k].plot([lo, hi], [lo + band, hi + band], "k--", linewidth = 0.7,
                   label = "+/-10% of range")
        ax[k].plot([lo, hi], [lo - band, hi - band], "k--", linewidth = 0.7)
        ax[k].set_xlabel(f"Actual {label} [{unit}]" if unit else f"Actual {label}")
        ax[k].set_ylabel(f"Predicted {label} [{unit}]" if unit else f"Predicted {label}")
        ax[k].set_aspect("equal", adjustable = "box")
        ax[k].legend(loc = "upper left", fontsize = 8)
        fig.colorbar(sc, ax = ax[k], label = "maximum camber, m")

    fig.suptitle(f"Parity -- {len(cases)} cases", fontsize = 10)
    fig.tight_layout()
    out = FIG_DIR / "parity.png"
    fig.savefig(out, dpi = DPI, bbox_inches = "tight")
    plt.close(fig)
    print(f"[plot] {out.name}")


def plot_history() -> None:
    "Training and validation loss against epoch"
    path = RUN_DIR / "history.csv"
    if not path.exists():
        print(f"[plot] no history.csv in {RUN_DIR}, skipping")
        return

    epochs, train, val, test, lr = [], [], [], [], []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            epochs.append(int(row["epoch"]))
            train.append(float(row["train_loss"]))
            val.append(float(row["val_loss"]))
            test.append(float(row["test_loss"]))
            lr.append(float(row["lr"]))

    fig, ax = plt.subplots(figsize = (8, 5))
    ax.semilogy(epochs, train, linewidth = 0.8, label = "training")
    ax.semilogy(epochs, val, linewidth = 0.8, label = "validation")
    if not np.all(np.isnan(test)):
        ax.semilogy(epochs, test, linewidth = 0.8, label = "test")

    best = int(np.nanargmin(val))
    ax.axvline(epochs[best], color = "k", linestyle = ":", linewidth = 0.8)
    ax.annotate(f"best epoch {epochs[best]}\nval = {val[best]:.3e}",
                xy = (epochs[best], val[best]), xytext = (10, 20),
                textcoords = "offset points", fontsize = 8)

    # mark the learning-rate decay steps
    for i in range(1, len(lr)):
        if lr[i] < lr[i - 1]:
            ax.axvline(epochs[i], color = "grey", linestyle = "--", linewidth = 0.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, which = "both", alpha = 0.2)
    fig.tight_layout()
    out = FIG_DIR / "history.png"
    fig.savefig(out, dpi = DPI, bbox_inches = "tight")
    plt.close(fig)
    print(f"[plot] {out.name}  (dashed grey lines are LR decay steps)")


def main() -> None:
    FIG_DIR.mkdir(parents = True, exist_ok = True)

    paths = sorted(FIELD_DIR.glob("dp*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"no .npz files in {FIELD_DIR} -- run evaluate.py with EXPORT_CASES > 0"
        )
    cases = [load_case(p) for p in paths]
    print(f"[plot] {len(cases)} case(s) from {FIELD_DIR}")

    for case in cases:
        for k in range(len(case["target_columns"])):
            plot_contours(case, k)
            plot_error(case, k)

    plot_parity(cases)
    plot_history()
    print(f"[plot] figures in {FIG_DIR}")


if __name__ == "__main__":
    main()