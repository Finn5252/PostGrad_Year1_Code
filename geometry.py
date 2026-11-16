# libraries

from __future__ import annotations

import numpy as np

# settings

CHORD = 1.0 # reference length for the coefficients

LEADING_EDGE_X = 0.0 # position of the leading edge in the CFD domain
LEADING_EDGE_Y = 1.0

N_PANELS = 200 # outline points per surface
QUERY_OFFSET = 1e-4 # how far outside the surface to look for the adjacent cell
MATCH_RADIUS = 50 * QUERY_OFFSET # a panel with no cell centre within this distance is left out


def naca4_profile(m: float, p: float, t: float, n: int = N_PANELS) -> np.ndarray:
    "Closed NACA 4-digit outline, upper surface then lower, as (2n, 2) coordinates"
    # cosine spacing clusters the points at the leading and trailing edges
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))

    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                    + 0.2843 * x**3 - 0.1015 * x**4)

    if m == 0.0 or p == 0.0:
        yc = np.zeros_like(x) # symmetric section
        dyc = np.zeros_like(x)
    else:
        fore = x <= p # ahead of the maximum camber position
        yc = np.where(fore,
                      m / p**2 * (2 * p * x - x**2),
                      m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
        dyc = np.where(fore,
                       2 * m / p**2 * (p - x),
                       2 * m / (1 - p)**2 * (p - x))

    # thickness added normal to the camber line
    theta = np.arctan(dyc)
    xu, yu = x - yt * np.sin(theta), yc + yt * np.cos(theta)
    xl, yl = x + yt * np.sin(theta), yc - yt * np.cos(theta)

    # lower surface reversed, so the outline runs once around the foil
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
    "Segment midpoints, outward unit normals and lengths"
    a = outline
    b = np.roll(outline, -1, axis = 0) # each point's successor, the last one wrapping back to the first
    d = b - a
    length = np.linalg.norm(d, axis = 1)

    keep = length > 1e-12 # drop zero-length segments, e.g. a sharp TE
    a, b, d, length = a[keep], b[keep], d[keep], length[keep]

    # the sign of the enclosed area gives the direction of travel, which decides which side is outward
    signed_area = 0.5 * np.sum(a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1])
    if signed_area > 0: # anticlockwise
        normal = np.stack([d[:, 1], -d[:, 0]], axis = 1)
    else: # clockwise
        normal = np.stack([-d[:, 1], d[:, 0]], axis = 1)
    normal /= length[:, None]

    return 0.5 * (a + b), normal, length
