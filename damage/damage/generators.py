"""
The four tissue-damage operators + the ``apply_damage`` entry point.

Design contract (why this is measurable):
  Every operator returns, for each *surviving* spot, the exact displacement from
  its pre-damage location. Registration ground truth is therefore the inverse of
  that displacement - a method that perfectly undoes the damage scores zero
  error. Removed spots (tissue loss, the blade path of a tear) are reported in a
  mask so partial-overlap methods can be scored only on recoverable spots.

  Damage is a pure function of (coords, damage_type, severity, seed): the same
  triple reproduces the same section exactly. Severity is an integer level
  0..5, where 0 is the undamaged control and 1..5 are increasing damage. Each
  level maps to an interpretable physical parameter (tear gap in spot-pitches,
  fraction of area lost, fold-flap fraction, stretch amplitude in pitches).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import (extent, median_pitch, principal_axes, signed_distance_to_polyline,
                   unit, unit_rows)
from .paths import make_path

DAMAGE_TYPES = ("tear", "tissue_loss", "fold", "stretch")
N_LEVELS = 5                      # severity levels 1..5 (+ level 0 control)

# Per-type severity schedule, index 0..5 (0 = identity control). Units noted.
SEVERITY = {
    "tear":        dict(gap_pitch=[0.0, 1.5, 3.0, 5.0, 8.0, 12.0]),     # gap opened, spot-pitches
    "tissue_loss": dict(area_frac=[0.0, 0.05, 0.10, 0.20, 0.35, 0.50]),  # fraction of spots removed
    "fold":        dict(flap_frac=[0.0, 0.08, 0.15, 0.25, 0.35, 0.45]),  # fraction of extent folded over
    "stretch":     dict(peak_pitch=[0.0, 2.0, 4.0, 7.0, 11.0, 16.0]),   # peak local displacement, spot-pitches
    # secondary: boundary-reaching tear (slit opens the outline); same gap schedule
    "tear_edge":   dict(gap_pitch=[0.0, 1.5, 3.0, 5.0, 8.0, 12.0]),
}
TEAR_PATHS = ("straight", "curved", "branching")
PRIMARY_TYPES = ("tear", "tissue_loss", "fold", "stretch")
SECONDARY_TYPES = ("tear_edge",)


@dataclass
class DamageResult:
    """Everything needed to score a registration against exact ground truth."""
    coords: np.ndarray            # (m,2) damaged coords of surviving spots
    original_coords: np.ndarray   # (m,2) pre-damage coords of survivors (GT target)
    displacement: np.ndarray      # (m,2) coords - original_coords (GT forward map)
    survivor_index: np.ndarray    # (m,) indices into the input spots that survived
    removed_index: np.ndarray     # (r,) indices removed from the input
    mask: np.ndarray              # (n_input,) per-spot label: see LABELS
    meta: dict = field(default_factory=dict)

    @property
    def n_input(self) -> int:
        return len(self.mask)

    @property
    def n_survivors(self) -> int:
        return len(self.survivor_index)


LABELS = ("intact", "removed", "displaced", "folded", "stretched")


# --------------------------------------------------------------------------- #
# operators: each returns (new_coords[n,2], removed[n]bool, moved_label[n]str-ish, meta)
# operating on the FULL input set (removal handled by the removed mask).
# --------------------------------------------------------------------------- #
def _tear(coords, pitch, ext, rng, *, gap_pitch, kind, cut_halfwidth_pitch=0.6,
          boundary=False):
    paths = make_path(coords, "edge" if boundary else kind, rng)
    n = len(coords)
    best = np.full(n, np.inf)
    side = np.ones(n)
    normal = np.zeros((n, 2))
    for path in paths:
        d, s, nrm = signed_distance_to_polyline(coords, path)
        take = d < best
        best[take] = d[take]
        side[take] = s[take]
        normal[take] = nrm[take]
    # thin blade path -> tissue actually cut away
    cut_hw = cut_halfwidth_pitch * pitch
    removed = best < cut_hw
    # Each edge retracts along its LOCAL cut normal, so the two lips separate
    # perpendicular to the cut everywhere along the path (works for curved and
    # branching paths without the two halves crossing over). The displacement is
    # a clean, invertible, piecewise map -> exact registration ground truth.
    gap = gap_pitch * pitch
    disp = np.zeros((n, 2))
    move = ~removed
    if boundary:
        # boundary-reaching tear: the path terminates on the section edge and only
        # ONE lip retracts (by the full gap), so a slit opens the outline itself
        # rather than an internal crack in a closed plate.
        openside = move & (side > 0)
        disp[openside] = gap * normal[openside]
    else:
        disp[move] = side[move, None] * (gap / 2.0) * normal[move]
    label = np.where(removed, "removed",
                     np.where(np.linalg.norm(disp, axis=1) > 1e-9, "displaced", "intact"))
    meta = dict(path_kind="edge" if boundary else kind, n_paths=len(paths),
                gap_px=float(gap), gap_pitch=float(gap_pitch),
                cut_halfwidth_px=float(cut_hw), boundary=bool(boundary),
                n_removed=int(removed.sum()))
    return coords + disp, removed, label, meta


def _tear_edge(coords, pitch, ext, rng, *, gap_pitch, kind="edge"):
    return _tear(coords, pitch, ext, rng, gap_pitch=gap_pitch, kind="edge",
                 boundary=True)


def _tissue_loss(coords, pitch, ext, rng, *, area_frac):
    n = len(coords)
    k = int(round(area_frac * n))
    removed = np.zeros(n, bool)
    if k > 0:
        # seed at a boundary (extreme spot along a random direction) so the lost
        # region opens from an edge, as real tissue loss does.
        d0 = unit(rng.normal(size=2))
        seed = coords[np.argmax(coords @ d0)]
        rel = coords - seed
        dist = np.linalg.norm(rel, axis=1)
        ang = np.arctan2(rel[:, 1], rel[:, 0])
        # smooth angular modulation -> organic (non-circular) boundary
        m = (1.0
             + 0.30 * np.sin(ang * rng.integers(2, 4) + rng.uniform(0, 6.28))
             + 0.20 * np.cos(ang * rng.integers(1, 3) + rng.uniform(0, 6.28)))
        score = dist * m
        removed[np.argsort(score)[:k]] = True
    label = np.where(removed, "removed", "intact")
    meta = dict(area_frac_target=float(area_frac),
                area_frac_achieved=float(removed.mean()), n_removed=int(removed.sum()))
    return coords.copy(), removed, label, meta


def _fold(coords, pitch, ext, rng, *, flap_frac):
    n = len(coords)
    c, U = principal_axes(coords)
    axis = U[:, int(rng.integers(2))]                 # fold-normal axis
    axis = unit(axis)
    proj = (coords - c) @ axis
    lo, hi = proj.min(), proj.max()
    span = hi - lo + 1e-9
    edge = 1 if rng.random() < 0.5 else -1
    if edge > 0:
        line = hi - flap_frac * span
        flap = proj > line
    else:
        line = lo + flap_frac * span
        flap = proj < line
    new_proj = 2.0 * line - proj                       # reflection across fold line
    disp = np.zeros((n, 2))
    disp[flap] = ((new_proj - proj)[flap])[:, None] * axis[None]
    label = np.where(flap, "folded", "intact")
    meta = dict(flap_frac=float(flap_frac), fold_axis=axis.tolist(),
                n_folded=int(flap.sum()))
    return coords + disp, np.zeros(n, bool), label, meta


def _stretch(coords, pitch, ext, rng, *, peak_pitch):
    n = len(coords)
    d0 = unit(rng.normal(size=2))
    center = coords[np.argmax(coords @ d0)]            # a boundary anchor
    sigma = 0.20 * ext
    r2 = ((coords - center) ** 2).sum(1)
    w = np.exp(-r2 / (2 * sigma ** 2))
    rdir = unit_rows(coords - center)                  # radial outward stretch
    disp = (peak_pitch * pitch) * w[:, None] * rdir
    label = np.where(w > 0.05, "stretched", "intact")
    meta = dict(peak_px=float(peak_pitch * pitch), peak_pitch=float(peak_pitch),
                sigma_px=float(sigma), center=center.tolist(),
                n_stretched=int((w > 0.05).sum()))
    return coords + disp, np.zeros(n, bool), label, meta


_OPS = {"tear": _tear, "tissue_loss": _tissue_loss, "fold": _fold,
        "stretch": _stretch, "tear_edge": _tear_edge}


def _params_for(dtype, level, rng, tear_path=None):
    sched = SEVERITY[dtype]
    p = {k: v[level] for k, v in sched.items()}
    if dtype == "tear":
        p["kind"] = tear_path or TEAR_PATHS[int(rng.integers(len(TEAR_PATHS)))]
    return p


def apply_damage(coords, damage_type, severity, seed, *, tear_path=None):
    """Damage a spot point cloud reproducibly.

    Parameters
    ----------
    coords : (n,2) array of spot coordinates (native units, e.g. Visium pixels).
    damage_type : one of DAMAGE_TYPES.
    severity : int level 0..5 (0 = undamaged control).
    seed : int; (damage_type, severity, seed) fully determines the output.
    tear_path : optional {"straight","curved","branching"} to force a tear path
        geometry; if None a geometry is chosen from the seed.

    Returns
    -------
    DamageResult
    """
    coords = np.asarray(coords, float)
    if damage_type not in _OPS:
        raise ValueError(f"unknown damage_type {damage_type!r}; pick from {DAMAGE_TYPES}")
    if not (0 <= severity <= N_LEVELS):
        raise ValueError(f"severity must be 0..{N_LEVELS}, got {severity}")
    rng = np.random.default_rng((hash((damage_type, int(severity))) & 0xFFFFFFFF, int(seed)))
    pitch = median_pitch(coords)
    ext = extent(coords)

    if severity == 0:                                   # undamaged control
        n = len(coords)
        return DamageResult(
            coords=coords.copy(), original_coords=coords.copy(),
            displacement=np.zeros((n, 2)), survivor_index=np.arange(n),
            removed_index=np.array([], int), mask=np.array(["intact"] * n, object),
            meta=dict(damage_type=damage_type, severity=0, seed=int(seed),
                      pitch=pitch, extent=ext, control=True))

    params = _params_for(damage_type, severity, rng, tear_path=tear_path)
    new_coords, removed, label, op_meta = _OPS[damage_type](
        coords, pitch, ext, rng, **params)

    survivor = np.where(~removed)[0]
    removed_idx = np.where(removed)[0]
    meta = dict(damage_type=damage_type, severity=int(severity), seed=int(seed),
                pitch=float(pitch), extent=float(ext), control=False, **op_meta)
    return DamageResult(
        coords=new_coords[survivor], original_coords=coords[survivor],
        displacement=(new_coords - coords)[survivor], survivor_index=survivor,
        removed_index=removed_idx, mask=np.asarray(label, object), meta=meta)
