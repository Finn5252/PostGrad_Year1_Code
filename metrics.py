from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# loss

class HalfMeanSquaredError(nn.Module):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
        return 0.5 * (pred - target).pow(2).mean()
class MeanAccumulator:
    "Unweighted mean over cases"
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self,count += n

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")

# relative L1/L2 error

@dataclass
class RelativeErrorConfig:
    "Handling nodes whose ground truth value approx. 0"

    mode: Literal["mask", "floor"] = "mask"
    threshold_mode: Literal["relative_rms", "absolute"] = "relative_rms"
    threshold: float = 1e-3

    def describe(self) -> str:
        unit = "x fielf RMS" if self.threshold_mode == "relative_rms" else "(absolute)"
        return f"{self.mode}, |phi| < {self.threshold:g} {unit}"

class RelativeErrorAccumulator:
    "2-pass accumulator for the relative errors per target field"

    def __init__(
            self,
            field_names: Sequence[str],
            cfg: Optional[RelativeErrorConfig] = None,
    ) -> None:
        self.field_names = list (field_names)
        self.cfg = cfg or RelativeErrorConfig()
        d = len(self.field_names)
        self._sq_sum = np.zeros(d)
        self._sq_count = np.zeros(d, dtype = np.int64)
        self._l1_sum = np.zeros(d)
        self._l2_sum = np.zeros(d)
        self._used = np.zeros(d, dtype = np.int64)
        self._total = np.zeros(d, dtype = np.int64)
        self._scale_locked = False
        self._thresholds: Optional[np.ndarray] = None

    # pass 1: field scale
 
    def update_scale(self, target: Tensor | np.ndarray) -> None:
        t = _to_numpy(target)
        self._sq_sum += (t**2).sum(axis = 0)
        self._sq_count += t.shape[0]
 
    def lock_scale(self) -> None:
        cfg = self.cfg
        if cfg.threshold_mode == "absolute":
            self._thresholds = np.full(len(self.field_names), float(cfg.threshold))
        else:
            with np.errstate(invalid = "ignore", divide = "ignore"):
                rms = np.sqrt(self._sq_sum / np.maximum(self._sq_count, 1))
            self._thresholds = cfg.threshold * rms
        self._scale_locked = True
 
    # pass 2: errors
 
    def update(self, pred: Tensor | np.ndarray, target: Tensor | np.ndarray) -> None:
        if not self._scale_locked:
            raise RuntimeError("call lock_scale() before update()")
        p, t = _to_numpy(pred), _to_numpy(target)
        if p.shape != t.shape:
            raise ValueError(f"shape mismatch: {p.shape} vs {t.shape}")
        if p.shape[1] != len(self.field_names):
            raise ValueError(f"expected {len(self.field_names)} fields, got {p.shape[1]}")
 
        diff = p - t
        mag = np.abs(t)
        self._total += p.shape[0]
 
        for j in range(p.shape[1]):
            thr = self._thresholds[j]
            if self.cfg.mode == "mask":
                keep = mag[:, j] > thr
                denom = mag[keep, j]
                num = np.abs(diff[keep, j])
            else:   # floor
                denom = np.maximum(mag[:, j], thr)
                num = np.abs(diff[:, j])
            ratio = num / denom
            self._l1_sum[j] += ratio.sum()
            self._l2_sum[j] += (ratio**2).sum()
            self._used[j] += int(ratio.size)
 
    # results
 
    def result(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for j, name in enumerate(self.field_names):
            n = int(self._used[j])
            total = int(self._total[j])
            if n == 0:
                l1 = l2 = float("nan")
            else:
                l1 = self._l1_sum[j] / n
                l2 = float(np.sqrt(self._l2_sum[j] / n))
            out[name] = {
                "L1": float(l1),
                "L2": float(l2),
                "n_nodes_used": n,
                "n_nodes_total": total,
                "excluded_fraction": (total - n) / total if total else 0.0,
                "zero_threshold": float(self._thresholds[j]),
                "handling": self.cfg.describe(),
            }
        return out
 
def _to_numpy(a) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    a = np.asarray(a, dtype = np.float64)
    return a if a.ndim == 2 else a.reshape(-1, 1)