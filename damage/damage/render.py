"""Matplotlib rendering of a DamageResult.

Two views:
  plot_reference  - the undamaged section at original coordinates, coloured by
                    what damage WILL do to each spot (removed / displaced / ...),
                    i.e. the damage mask overlaid on the reference.
  plot_damaged    - the damaged section at its new coordinates, same colouring.

Put side by side you can see exactly which spots were removed, which moved, and
where they went.
"""
from __future__ import annotations

import numpy as np

LABEL_COLORS = {
    "intact": "#c2c7cc",
    "removed": "#d1495b",
    "displaced": "#2f6fed",
    "folded": "#f6a609",
    "stretched": "#12a150",
}
LABEL_ORDER = ("intact", "removed", "displaced", "folded", "stretched")


def _style(ax):
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])


def plot_reference(full_coords, result, ax, *, point_size=3, title=None):
    """Undamaged section (all input spots) coloured by the damage mask - the
    reference with the damage preview overlaid."""
    full_coords = np.asarray(full_coords, float)
    mask = result.mask
    for lab in LABEL_ORDER:
        m = mask == lab
        if m.any():
            ax.scatter(full_coords[m, 0], full_coords[m, 1], s=point_size,
                       c=LABEL_COLORS[lab], linewidths=0)
    _style(ax)
    if title:
        ax.set_title(title, fontsize=8)


def plot_damaged(result, ax, *, point_size=3, title=None, ref_ghost=None):
    """Damaged section at new coordinates, coloured by surviving-spot mask.
    Optionally underlays a faint ghost of the reference footprint."""
    if ref_ghost is not None:
        g = np.asarray(ref_ghost, float)
        ax.scatter(g[:, 0], g[:, 1], s=point_size, c="#eef0f2", linewidths=0, zorder=0)
    surv = result.survivor_index
    lab_surv = result.mask[surv]
    for lab in LABEL_ORDER:
        m = lab_surv == lab
        if m.any():
            ax.scatter(result.coords[m, 0], result.coords[m, 1], s=point_size,
                       c=LABEL_COLORS[lab], linewidths=0, zorder=2)
    _style(ax)
    if title:
        ax.set_title(title, fontsize=8)


def plot_pair(full_coords, result, axes, *, point_size=3, tag=""):
    """[reference | damaged] side-by-side pair on two axes."""
    plot_reference(full_coords, result, axes[0], point_size=point_size,
                   title=f"reference {tag}".strip())
    plot_damaged(result, axes[1], point_size=point_size,
                 title=f"damaged {tag}".strip(), ref_ghost=full_coords)
    return axes


# legacy alias kept for the package API
def plot_damage(result, ax=None, *, point_size=4, show_original=True, title=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    plot_damaged(result, ax, point_size=point_size, title=title)
    return ax
