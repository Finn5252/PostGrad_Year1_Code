# libraries

from __future__ import annotations

import hashlib
import json

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor

from data import CaseStore, DataConfig, ScalerBundle, targets_to_physical
from geometry import (
    CHORD, LEADING_EDGE_X, LEADING_EDGE_Y, MATCH_RADIUS, N_PANELS, QUERY_OFFSET, foil_outline, panels,
)
from settings import RHO

SURFACE_LOSSES = (None, "drag", "cp") # None turns the surface term off
WEIGHTING_MODES = ("fixed", "cov") # constant weights, or weights from each loss's coefficient of variation
GEOMETRY_COLUMNS = ("m", "p", "t", "AoA") # the scalars needed to rebuild the foil outline
PANEL_VERSION = "naca4-panels-v1" # part of the cache key, change it whenever the panel construction changes


# panel geometry

class PanelCache:
    "On-disk cache of the surface panels and the cell adjacent to each, keyed like GraphCache"

    def __init__(self, cfg: DataConfig, store: CaseStore) -> None:
        names = list(cfg.scalar_columns)
        missing = [n for n in GEOMETRY_COLUMNS if n not in names]
        if missing:
            raise KeyError(f"the surface losses need scalar columns {missing} to rebuild the foil")
        self.geometry_pos = [names.index(n) for n in GEOMETRY_COLUMNS]

        # everything that affects the panels goes into the hash, as for the KNN cache
        payload = {
            "graphs": cfg.cache_key(store.fingerprint),
            "n_panels": N_PANELS,
            "query_offset": QUERY_OFFSET,
            "match_radius": MATCH_RADIUS,
            "leading_edge": [LEADING_EDGE_X, LEADING_EDGE_Y],
            "version": PANEL_VERSION,
        }
        blob = json.dumps(payload, sort_keys = True).encode()
        self.key = hashlib.sha256(blob).hexdigest()[:16]
        self.path = Path(cfg.cache_dir) / f"panels_{self.key}.npz"
        self.store = store
        self._arrays: Optional[dict[str, np.ndarray]] = None

    def build_or_load(self, verbose: bool = True) -> None:
        if self.path.exists():
            with np.load(self.path) as z:
                self._arrays = {name: z[name] for name in z.files}
            if verbose:
                print(f"[data] loaded cached surface panels from {self.path}")
        else:
            self._build(verbose)

        if len(self._arrays["offsets"]) != self.store.n_cases + 1:
            raise RuntimeError(f"panel cache {self.path} does not match the store; delete it to rebuild")

        frac = self.matched_fraction
        if verbose:
            worst = int(frac.argmin())
            print(
                f"[data] panels matched to a cell: {frac.mean():.2%} "
                f"(worst case {frac[worst]:.2%}, dp_id={int(self.store.dp_ids[worst])})"
            )
        empty = np.flatnonzero(frac == 0)
        if empty.size:
            raise RuntimeError(
                f"{empty.size} case(s) have no panel matched to a cell, first is case {int(empty[0])} "
                f"-- the foil geometry and the mesh do not line up"
            )

    def _build(self, verbose: bool) -> None:
        if verbose:
            print(f"[data] building surface panels ({N_PANELS} points per surface) -> {self.path}")
        parts = {name: [] for name in ("mid", "normal", "ds", "cell", "matched")}
        offsets = [0]
        for i in range(self.store.n_cases):
            nodes, _, scalars, _ = self.store.cropped_case(i)
            xy = nodes[:, [self.store.x_pos, self.store.y_pos]]
            outline = foil_outline(*(float(scalars[j]) for j in self.geometry_pos))
            mid, normal, ds = panels(outline)
            # the cell adjacent to a panel is the one nearest to a point just outside it
            dist, cell = cKDTree(xy).query(mid + QUERY_OFFSET * normal)
            for name, value in zip(parts, (mid, normal, ds, cell, dist < MATCH_RADIUS)):
                parts[name].append(value)
            offsets.append(offsets[-1] + len(ds))
            if verbose and (i + 1) % 100 == 0:
                print(f"[data]   {i + 1}/{self.store.n_cases} cases")

        arrays = {name: np.concatenate(values) for name, values in parts.items()}
        arrays["cell"] = arrays["cell"].astype(np.int64)
        arrays["offsets"] = np.asarray(offsets, dtype = np.int64)
        self.path.parent.mkdir(parents = True, exist_ok = True)
        np.savez_compressed(self.path, **arrays)
        self._arrays = arrays

    @property
    def matched_fraction(self) -> np.ndarray:
        "Fraction of each case's panels that found an adjacent cell"
        offsets = self._arrays["offsets"]
        # reduceat sums the matched flags between consecutive offsets, i.e. per case
        hits = np.add.reduceat(self._arrays["matched"].astype(np.int64), offsets[:-1])
        return hits / np.diff(offsets)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        "Adjacent cell, n_x * ds and ds of every matched panel of case i"
        if self._arrays is None:
            self.build_or_load()
        a, b = int(self._arrays["offsets"][i]), int(self._arrays["offsets"][i + 1])
        keep = self._arrays["matched"][a:b]
        ds = self._arrays["ds"][a:b]
        nx_ds = self._arrays["normal"][a:b, 0] * ds
        return self._arrays["cell"][a:b][keep], nx_ds[keep], ds[keep]


# force integration

def pressure_drag_coefficients(
        pressure: Tensor, # (N,) Pa, every node in the batch
        panel_index: Tensor, # (P,) adjacent node of each panel
        panel_nx_ds: Tensor, # (P,) outward normal x component times panel length
        node_batch: Tensor, # (N,) which case each node belongs to
        v_in: Tensor, # (B,) inlet velocity per case
        n_graphs: int,
) -> Tensor:
    "Cdp per case: F_x = -sum p n_x ds over 0.5 rho V^2 c"
    contrib = -pressure[panel_index] * panel_nx_ds
    # the case comes from the node a panel points at, so a wrong offset puts it in the wrong case
    case = node_batch[panel_index]
    drag = contrib.new_zeros(n_graphs).index_add(0, case, contrib) # panel contributions summed per case
    return drag / (0.5 * RHO * v_in**2 * CHORD)


class _SurfaceLoss:
    "What the two surface losses share: the scalers and the pressure in Pa"

    def __init__(self, scalers: ScalerBundle) -> None:
        if scalers.target_normalisation != "none" and scalers.rho != RHO:
            raise ValueError(f"scalers use rho={scalers.rho} but settings.py has RHO={RHO}")
        self.scalers = scalers
        self.i_pressure = list(scalers.target_columns).index("static_pressure")

    def pressure(self, z: Tensor, batch) -> Tensor:
        "Pressure in Pa at every node, from scaled targets or predictions"
        # double precision, since the forces are small differences of large pressures
        return targets_to_physical(z.double(), batch, self.scalers)[:, self.i_pressure]


class DragLoss(_SurfaceLoss):
    "Absolute error in pressure drag coefficient, per case, averaged over the cases in the batch"

    def coefficients(self, z: Tensor, batch) -> Tensor:
        "Cdp per case from scaled targets or predictions, shape (num_graphs,)"
        pressure = self.pressure(z, batch)
        v_in = batch.v_in.to(pressure) # raw, one per case
        return pressure_drag_coefficients(
            pressure, batch.panel_index, batch.panel_nx_ds, batch.batch, v_in, int(batch.num_graphs)
        )

    def __call__(self, pred: Tensor, batch) -> Tensor:
        cd_pred = self.coefficients(pred, batch)
        with torch.no_grad():
            cd_true = self.coefficients(batch.y, batch)
        return (cd_pred - cd_true).abs().mean().to(pred.dtype)


class SurfaceCpLoss(_SurfaceLoss):
    """Squared Cp error at the cell next to each surface panel, length-weighted per case
    (sum ds e^2 / sum ds), averaged over the cases in the batch. Same panels as DragLoss"""

    def surface_cp(self, z: Tensor, batch) -> Tensor:
        "Cp at every matched panel in the batch, from scaled targets or predictions"
        pressure = self.pressure(z, batch)
        case = batch.batch[batch.panel_index]
        q = 0.5 * RHO * batch.v_in.to(pressure)**2 # one per case
        return pressure[batch.panel_index] / q[case]

    def per_case(self, pred: Tensor, batch) -> Tensor:
        "Length-weighted mean squared Cp error of each case, shape (num_graphs,)"
        with torch.no_grad():
            cp_true = self.surface_cp(batch.y, batch)
        err2 = (self.surface_cp(pred, batch) - cp_true)**2
        ds = batch.panel_ds
        case = batch.batch[batch.panel_index]
        n = int(batch.num_graphs)
        num = err2.new_zeros(n).index_add(0, case, ds * err2)
        den = ds.new_zeros(n).index_add(0, case, ds)
        return num / den

    def __call__(self, pred: Tensor, batch) -> Tensor:
        return self.per_case(pred, batch).mean().to(pred.dtype)


# weighting of the field loss against the surface loss

class FixedWeighting:
    "Constant coefficients (field, surface) on the raw losses"

    def __init__(self, reference: tuple[float, float]) -> None:
        self.reference = reference

    def weights(self) -> tuple[float, float]:
        return self.reference

    def observe(self, field: float, surface: float) -> None:
        pass # nothing to track, present so both weightings are used the same way

    def summary(self) -> dict:
        return {"mode": "fixed", "weights": list(self.reference)}


class CoVWeighting:
    """Coefficient-of-variation weighting of (field, surface). Each loss is divided by its own running
    mean, and the loss whose ratio varies more, relative to its mean, gets the larger weight"""

    def __init__(
            self, reference: tuple[float, float], surface_scale: float, window: int, warmup: int,
    ) -> None:
        self.reference = reference # used for the logged loss and for model selection only
        self.surface_scale = surface_scale # field loss reference / surface loss reference
        self.window = window
        self.warmup = warmup
        self.step = 0
        self.smoothed = np.zeros(2) # (field, surface) averaged over the last window steps
        self.n = 0 # observations fed to the CoV statistics
        self.mean_L = np.zeros(2) # running mean of the smoothed losses
        self.mean_l = np.zeros(2) # running mean of the loss ratios, smoothed / mean_L
        self.M_l = np.zeros(2) # running variance of the loss ratios

    def alphas(self) -> tuple[float, float]:
        "Weights summing to 1"
        if self.step < self.warmup or self.n < 2:
            return (0.5, 0.5)
        c = np.sqrt(self.M_l) / self.mean_l # coefficient of variation of each loss ratio
        # equal weights if a statistic is not usable yet, e.g. a loss that has not varied
        if not np.all(np.isfinite(c)) or c.sum() <= 0:
            return (0.5, 0.5)
        alpha = c / c.sum()
        return (float(alpha[0]), float(alpha[1]))

    def weights(self) -> tuple[float, float]:
        "Coefficients on the raw losses"
        a_field, a_surface = self.alphas()
        return (a_field, a_surface * self.surface_scale)

    def observe(self, field: float, surface: float) -> None:
        self.step += 1
        self.smoothed += (np.array([field, surface]) - self.smoothed) / min(self.step, self.window)
        # statistics start once the window has filled, so early single-case values never enter them
        if self.step < self.window:
            return
        self.n += 1
        t = self.n
        ratio = self.smoothed / self.mean_L if t > 1 else np.ones(2) # no mean to divide by on the first observation
        self.mean_L += (self.smoothed - self.mean_L) / t
        previous = self.mean_l.copy()
        self.mean_l += (ratio - self.mean_l) / t
        # variance updated in place from the mean before and after this observation
        self.M_l = (1 - 1 / t) * self.M_l + (ratio - previous) * (ratio - self.mean_l) / t

    def summary(self) -> dict:
        return {
            "mode": "cov",
            "surface_scale": self.surface_scale,
            "window_steps": self.window,
            "warmup_steps": self.warmup,
            "reference_weights": list(self.reference),
        }
