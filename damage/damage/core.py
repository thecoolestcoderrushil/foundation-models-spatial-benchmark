"""
Geometry helpers for the damage generator.

Everything here is numpy-only and deterministic given an explicit
``numpy.random.Generator`` so a (damage-type, severity, seed) triple always
reproduces the same damaged section bit-for-bit.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def median_pitch(coords: np.ndarray) -> float:
    """Median nearest-neighbour distance - the spot-to-spot pitch (native units)."""
    coords = np.asarray(coords, float)
    if len(coords) < 2:
        return 1.0
    d, _ = cKDTree(coords).query(coords, k=2)
    return float(np.median(d[:, 1]))


def extent(coords: np.ndarray) -> float:
    """Diagonal size of the section's bounding box (native units)."""
    coords = np.asarray(coords, float)
    return float(np.linalg.norm(coords.max(0) - coords.min(0)))


def principal_axes(coords: np.ndarray):
    """Return (centroid, U) where U[:,0] is the section's long axis (unit vectors)."""
    coords = np.asarray(coords, float)
    c = coords.mean(0)
    _, _, Vt = np.linalg.svd(coords - c, full_matrices=False)
    return c, Vt.T  # columns are principal directions


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


# --------------------------------------------------------------------------- #
# Polyline / path geometry (used by tear + fold lines)
# --------------------------------------------------------------------------- #
def resample_polyline(pts: np.ndarray, n: int = 200) -> np.ndarray:
    """Arc-length resample a polyline to n points so distance queries are smooth."""
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return np.repeat(pts[:1], n, axis=0)
    u = np.linspace(0.0, s[-1], n)
    x = np.interp(u, s, pts[:, 0])
    y = np.interp(u, s, pts[:, 1])
    return np.column_stack([x, y])


def signed_distance_to_polyline(coords: np.ndarray, path: np.ndarray):
    """For each point: (unsigned distance to the polyline, signed side, nearest
    segment's outward normal at the closest point).

    Side sign is defined by the cross product with the local tangent, so points
    on the left of the path (in the path's travel direction) get +1. Returns
    (dist[n], side[n], normal[n,2]).
    """
    coords = np.asarray(coords, float)
    a = path[:-1]                      # segment starts (m,2)
    b = path[1:]                       # segment ends
    ab = b - a                         # (m,2)
    L2 = (ab ** 2).sum(1) + 1e-12
    # projection param t of each point onto each segment, clamped to [0,1]
    ap = coords[:, None, :] - a[None, :, :]          # (n,m,2)
    t = np.clip((ap * ab[None]).sum(2) / L2[None], 0.0, 1.0)   # (n,m)
    proj = a[None] + t[..., None] * ab[None]         # (n,m,2)
    d = np.linalg.norm(coords[:, None, :] - proj, axis=2)      # (n,m)
    j = d.argmin(1)                                  # nearest segment per point
    n_idx = np.arange(len(coords))
    dist = d[n_idx, j]
    tang = unit_rows(ab[j])                          # (n,2) tangent of nearest seg
    normal = np.column_stack([-tang[:, 1], tang[:, 0]])        # left normal
    rel = coords - proj[n_idx, j]
    side = np.sign((rel * normal).sum(1))
    side[side == 0] = 1.0
    return dist, side, normal


def unit_rows(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return v / n
