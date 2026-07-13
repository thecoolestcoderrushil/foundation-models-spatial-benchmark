"""
Tear-path generation.

A tear is a *cut that propagates along a path*, not a random blob. We generate a
path that crosses the section from one edge to the other and support three
geometries:

  * ``straight``   - a chord across the section at a random orientation.
  * ``curved``     - a smooth sinusoidal deviation about that chord (a wandering
                     cut).
  * ``branching``  - a main path plus one branch forking off it at a junction
                     (a Y-shaped tear).

All paths are returned as arc-length-resampled polylines in the section's native
coordinate units, generated deterministically from a numpy Generator.
"""
from __future__ import annotations

import numpy as np

from .core import extent, principal_axes, resample_polyline, rotation, unit


def _endpoints(coords, rng):
    """A chord across the section: a centre, a direction, and two far endpoints."""
    c, U = principal_axes(coords)
    # random orientation, biased toward crossing the short axis (a realistic cut
    # runs across the tissue, not along its length)
    theta = rng.uniform(0, np.pi)
    direction = rotation(theta) @ U[:, 1]
    half = 0.75 * extent(coords)
    offset = (rng.random() - 0.5) * 0.5 * extent(coords)
    perp = np.array([-direction[1], direction[0]])
    mid = c + perp * offset
    p0 = mid - direction * half
    p1 = mid + direction * half
    return p0, p1, unit(direction), perp


def straight_path(coords, rng):
    p0, p1, *_ = _endpoints(coords, rng)
    return resample_polyline(np.vstack([p0, p1]))


def curved_path(coords, rng, amplitude_frac=0.12, n_waves=None):
    """Chord with a sinusoidal lateral wobble - a wandering cut."""
    p0, p1, d, perp = _endpoints(coords, rng)
    ext = extent(coords)
    n_waves = n_waves if n_waves is not None else rng.uniform(0.75, 2.0)
    phase = rng.uniform(0, 2 * np.pi)
    amp = amplitude_frac * ext * rng.uniform(0.6, 1.4)
    ts = np.linspace(0, 1, 120)
    base = p0[None] + (p1 - p0)[None] * ts[:, None]
    wob = amp * np.sin(2 * np.pi * n_waves * ts + phase)
    pts = base + perp[None] * wob[:, None]
    return resample_polyline(pts)


def branching_path(coords, rng, amplitude_frac=0.08):
    """A main (slightly curved) path plus one branch forking at a junction."""
    main = curved_path(coords, rng, amplitude_frac=amplitude_frac)
    j = int(rng.uniform(0.35, 0.65) * len(main))     # junction along the main path
    junction = main[j]
    # branch heading: rotate the local main tangent by a sharp angle
    tang = main[min(j + 1, len(main) - 1)] - main[max(j - 1, 0)]
    ang = rng.choice([-1, 1]) * rng.uniform(np.pi / 5, np.pi / 2.5)
    bdir = rotation(ang) @ unit(tang)
    blen = rng.uniform(0.3, 0.55) * extent(coords)
    tip = junction + bdir * blen
    # gently curve the branch too
    perp = np.array([-bdir[1], bdir[0]])
    ts = np.linspace(0, 1, 60)
    base = junction[None] + (tip - junction)[None] * ts[:, None]
    wob = amplitude_frac * extent(coords) * rng.uniform(0.4, 1.0) \
        * np.sin(np.pi * ts + rng.uniform(0, np.pi))
    branch = base + perp[None] * wob[:, None]
    return main, resample_polyline(branch)


def edge_path(coords, rng):
    """A path that STARTS on the section boundary and terminates in the interior
    (for a boundary-reaching tear whose slit opens the outline itself)."""
    c, _ = principal_axes(coords)
    d0 = unit(rng.normal(size=2))
    start = coords[np.argmax(coords @ d0)]            # an extreme (boundary) spot
    inward = unit(c - start)
    inward = rotation(rng.uniform(-0.4, 0.4)) @ inward
    length = rng.uniform(0.45, 0.70) * extent(coords)
    tip = start + inward * length
    perp = np.array([-inward[1], inward[0]])
    ts = np.linspace(0, 1, 80)
    base = start[None] + (tip - start)[None] * ts[:, None]
    wob = 0.06 * extent(coords) * rng.uniform(0.3, 1.0) * np.sin(np.pi * ts + rng.uniform(0, np.pi))
    return resample_polyline(base + perp[None] * wob[:, None])


def make_path(coords, kind, rng):
    """Return a list of polylines (branching returns two, others return one)."""
    if kind == "edge":
        return [edge_path(coords, rng)]
    if kind == "straight":
        return [straight_path(coords, rng)]
    if kind == "curved":
        return [curved_path(coords, rng)]
    if kind == "branching":
        main, branch = branching_path(coords, rng)
        return [main, branch]
    raise ValueError(f"unknown tear path kind: {kind!r}")
