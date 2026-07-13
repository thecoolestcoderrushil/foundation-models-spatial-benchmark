"""Matplotlib rendering of a DamageResult (before / after with mask colouring)."""
from __future__ import annotations

import numpy as np

_LABEL_COLORS = {
    "intact": "#9aa0a6",
    "removed": "#d1495b",
    "displaced": "#3d5afe",
    "folded": "#f6a609",
    "stretched": "#12a150",
}


def plot_damage(result, ax=None, *, point_size=4, show_original=True, title=None):
    """Scatter the damaged section, coloured by damage label.

    Removed spots are drawn faintly at their original position (so tissue loss
    is visible); surviving spots are drawn at their damaged position.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    orig = result.original_coords
    surv = result.survivor_index
    mask = result.mask
    # ghost of removed tissue at original location
    rem = result.removed_index
    if show_original and len(rem):
        # removed spots' original coords aren't in result.original_coords (that is
        # survivors only); recover from meta-free path: caller passes full coords
        pass
    lab_surv = mask[surv]
    for lab in ("intact", "displaced", "folded", "stretched"):
        m = lab_surv == lab
        if m.any():
            ax.scatter(result.coords[m, 0], result.coords[m, 1], s=point_size,
                       c=_LABEL_COLORS[lab], linewidths=0, label=lab)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def plot_before_after(full_coords, result, axes=None, *, point_size=4, title=None):
    """Two panels: original section (with the doomed region highlighted) and the
    damaged section. `full_coords` is the pre-damage coordinate array."""
    import matplotlib.pyplot as plt
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(8, 4))
    full_coords = np.asarray(full_coords, float)

    # LEFT: original, colour the spots that will be removed / moved
    mask = result.mask
    col = np.array([_LABEL_COLORS.get(l, "#9aa0a6") for l in mask], object)
    axes[0].scatter(full_coords[:, 0], full_coords[:, 1], s=point_size,
                    c=list(col), linewidths=0)
    axes[0].set_title("original (damage preview)", fontsize=9)

    # RIGHT: damaged section
    plot_damage(result, ax=axes[1], point_size=point_size, title="damaged")
    for ax in axes:
        ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
    if title:
        axes[0].figure.suptitle(title, fontsize=10)
    return axes
