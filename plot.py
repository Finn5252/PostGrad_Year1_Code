
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
SMOOTH_SIGMA = 1.0      # gaussian smoothing of the interpolated grid
N_CONTOURS = 20

RHO = 1025.0 # fluid density, for the NMAPE reference

LEADING_EDGE_X = 0.0    # position of LE in CFD domain 
LEADING_EDGE_Y = 1.0
DRAW_FOIL = True       

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

    # Mask grid points with no nearby data
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


def naca4_profile(m: float, p: float, t: float, n: int = 200) -> np.ndarray:
    """Closed NACA 4-digit outline, upper surface then lower, as (2n, 2) coordinates.
    """
    # cosine spacing
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))

    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

    if m == 0.0 or p == 0.0:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)
    else:
        fore = x <= p
        yc = np.where(fore,
                      m / p**2 * (2 * p * x - x**2),
                      m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
        dyc = np.where(fore,
                       2 * m / p**2 * (p - x),
                       2 * m / (1 - p)**2 * (p - x))

    theta = np.arctan(dyc)
    xu, yu = x - yt * np.sin(theta), yc + yt * np.cos(theta)
    xl, yl = x + yt * np.sin(theta), yc - yt * np.cos(theta)

    return np.concatenate([
        np.stack([xu, yu], axis = 1),
        np.stack([xl[::-1], yl[::-1]], axis = 1),
    ])


def foil_outline(case: dict) -> np.ndarray:
    """The case's foil, rotated about the leading edge by the angle of attack and placed
    at the leading-edge position used in the CFD domain."""
    names = case["scalar_columns"]
    s = case["scalars"]
    m = float(s[names.index("m")])
    p = float(s[names.index("p")])
    t = float(s[names.index("t")])
    aoa = np.radians(float(s[names.index("AoA")]))

    xy = naca4_profile(m, p, t)
    c, sn = np.cos(-aoa), np.sin(-aoa)          # nose-up rotation about the leading edge
    rot = np.stack([xy[:, 0] * c - xy[:, 1] * sn,
                    xy[:, 0] * sn + xy[:, 1] * c], axis = 1)
    return rot + np.array([LEADING_EDGE_X, LEADING_EDGE_Y])


def draw_foil(ax, case: dict) -> None:
    xy = foil_outline(case)
    ax.fill(xy[:, 0], xy[:, 1], facecolor = "white", edgecolor = "k",
            linewidth = 0.8, zorder = 5)


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

    fig, ax = plt.subplots(2, 1, figsize = (9, 7), sharex = True, sharey = True)

    for a, Z, title in ((ax[0], ZP, "Predicted"), (ax[1], ZT, "Actual (CFD)")):
        cf = a.contourf(XI, YI, Z, levels = levels, cmap = "viridis", extend = "both")
        a.contour(XI, YI, Z, levels = levels, colors = "k", linewidths = 0.25)
        if DRAW_FOIL:
            draw_foil(a, case)
        a.set_title(title)
        a.set_ylabel("y")
        a.set_aspect("equal")
        fig.colorbar(cf, ax = a, label = f"{label} [{unit}]" if unit else label)

    # overlay
    ax[1].set_xlabel("x")

    fig.suptitle(f"{label} -- {case_title(case)}", fontsize = 10)
    fig.tight_layout()
    out = FIG_DIR / f"dp{case['dp_id']}_{name}_contours.png"
    fig.savefig(out, dpi = DPI, bbox_inches = "tight")
    plt.close(fig)
    print(f"[plot] {out.name}")


def plot_error(case: dict, field_index: int) -> None:
    "Signed difference, with the case NMAPE annotated"
    name = case["target_columns"][field_index]
    label, unit, scale = FIELD_LABELS.get(name, (name, "", 1.0))

    pred = case["pred"][:, field_index]
    true = case["true"][:, field_index]

    # NMAPE
    v_in = float(case["scalars"][case["scalar_columns"].index("V_in")])
    ref = v_in if name == "velocity_magnitude" else 0.5 * RHO * v_in**2
    nmape = 100.0 * float(np.abs(pred - true).mean() / ref)
    worst = 100.0 * float(np.abs(pred - true).max() / ref)

    diff = (pred - true) * scale
    span = np.abs(diff).max()

    fig, ax = plt.subplots(figsize = (9, 4))
    XI, YI, ZD = to_grid(case["x"], case["y"], diff)
    cf = ax.contourf(XI, YI, ZD, levels = np.linspace(-span, span, 41),
                     cmap = "RdBu_r", extend = "both")
    ax.set_title(f"Predicted - actual   NMAPE = {nmape:.3f}%   "
                 f"worst node = {worst:.2f}%")
    if DRAW_FOIL:
        draw_foil(ax, case)
    fig.colorbar(cf, ax = ax, label = f"difference [{unit}]" if unit else "difference")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.text(0.01, -0.28, f"max |error| = {span:.4g} {unit},  "
            f"normalised by {'V_in' if name == 'velocity_magnitude' else '0.5*rho*V_in^2'}"
            f" = {ref:.4g}",
            transform = ax.transAxes, fontsize = 8, color = "0.35")

    fig.suptitle(f"{label} error -- {case_title(case)}", fontsize = 10)
    fig.tight_layout()
    out = FIG_DIR / f"dp{case['dp_id']}_{name}_error.png"
    fig.savefig(out, dpi = DPI, bbox_inches = "tight")
    plt.close(fig)
    print(f"[plot] {out.name}  NMAPE={nmape:.3f}%")


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
        line = np.array([lo, hi])
        ax[k].plot(line, line, "r--", linewidth = 1, label = "1:1 line")
        # relative bands
        ax[k].plot(line, line * 1.1, "k--", linewidth = 0.7, label = "+10% error")
        ax[k].plot(line, line * 0.9, "k--", linewidth = 0.7, label = "-10% error")
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