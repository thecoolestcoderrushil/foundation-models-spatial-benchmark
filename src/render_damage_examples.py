"""
STOP-GATE deliverable: render damage examples for visual review BEFORE any sweep.

For a representative DLPFC section we render, per damage type, a grid over
severity (rows) x 3 examples (seeds). EACH example is a [reference | damaged]
pair, side by side, with the damage mask overlaid on both panels:
  reference panel  = undamaged spots, coloured by what damage will do to them
                     (grey intact, red removed, blue displaced, orange folded,
                     green stretched)
  damaged panel    = the result at new coordinates, faint grey ghost of the
                     original footprint underneath

Outputs -> results/damage_examples/
  <type>_pairs.png       severity x [3 side-by-side reference|damaged examples]
  tear_geometries.png    straight / curved / branching, each as a ref|damaged pair
  example_artifact.npz   emitted GT artifact (mask + displacement) for one case
  README.md

Usage:  python src/render_damage_examples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "damage"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import anndata as ad

from damage import apply_damage, DAMAGE_TYPES, SEVERITY, TEAR_PATHS
from damage.render import plot_reference, plot_damaged, LABEL_COLORS, LABEL_ORDER

DATA = Path(r"C:/Users/karti/arca/data")
SECTION = "DLPFC_151507"
OUT = ROOT / "results" / "damage_examples"
SEEDS = (0, 1, 2)
LEVELS = list(range(0, 6))          # 0 control .. 5 max


def load_section():
    A = ad.read_h5ad(DATA / f"{SECTION}.h5ad")
    return np.asarray(A.obsm["spatial"], float)


def sev_label(dtype, lvl):
    key, vals = next(iter(SEVERITY[dtype].items()))
    v = vals[lvl]
    if key == "area_frac":
        return f"sev {lvl}: {int(v*100)}% lost"
    if key == "flap_frac":
        return f"sev {lvl}: flap {int(v*100)}%"
    return f"sev {lvl}: {key.split('_')[0]} {v:g}"


def legend(fig):
    handles = [Patch(color=LABEL_COLORS[l], label=l) for l in LABEL_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=len(LABEL_ORDER),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))


def pairs_for_type(coords, dtype):
    nrow, ncol = len(LEVELS), 2 * len(SEEDS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.05 * nrow))
    for r, lvl in enumerate(LEVELS):
        for si, seed in enumerate(SEEDS):
            res = apply_damage(coords, dtype, lvl, seed=seed)
            a_ref, a_dam = axes[r, 2 * si], axes[r, 2 * si + 1]
            plot_reference(coords, res, a_ref, point_size=2.2,
                           title="reference" if r == 0 else None)
            plot_damaged(res, a_dam, point_size=2.2, ref_ghost=coords,
                         title="damaged" if r == 0 else None)
            if si == 0:
                a_ref.set_ylabel(sev_label(dtype, lvl), fontsize=8)
        # seed captions along the top
    for si, seed in enumerate(SEEDS):
        axes[0, 2 * si].annotate(f"example {si+1} (seed {seed})",
                                 xy=(1.0, 1.28), xycoords="axes fraction",
                                 ha="center", fontsize=9, annotation_clip=False)
    fig.suptitle(f"Damage type: {dtype}   (section {SECTION}, {len(coords)} spots)  "
                 "- reference | damaged pairs, mask overlaid", fontsize=12, y=1.005)
    legend(fig)
    fig.tight_layout()
    p = OUT / f"{dtype}_pairs.png"
    fig.savefig(p, dpi=115, bbox_inches="tight")
    plt.close(fig)
    return p


def tear_geometry_panel(coords):
    fig, axes = plt.subplots(len(TEAR_PATHS), 2, figsize=(7.5, 3.4 * len(TEAR_PATHS)))
    for i, kind in enumerate(TEAR_PATHS):
        res = apply_damage(coords, "tear", severity=4, seed=0, tear_path=kind)
        plot_reference(coords, res, axes[i, 0], point_size=2.4,
                       title=f"reference ({kind})")
        plot_damaged(res, axes[i, 1], point_size=2.4, ref_ghost=coords,
                     title=f"damaged ({kind})")
    fig.suptitle("Tear geometries at severity 4: straight / curved / branching",
                 fontsize=12, y=1.01)
    legend(fig)
    fig.tight_layout()
    p = OUT / "tear_geometries.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p


def emit_example_artifact(coords):
    res = apply_damage(coords, "tear", severity=3, seed=0, tear_path="curved")
    p = OUT / "example_artifact.npz"
    np.savez_compressed(
        p, input_coords=coords, damaged_coords=res.coords,
        original_coords=res.original_coords, gt_displacement=res.displacement,
        survivor_index=res.survivor_index, removed_index=res.removed_index,
        mask=res.mask.astype("U12"), meta=np.array(str(res.meta), dtype=object))
    return p, res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    coords = load_section()
    made = []
    for dtype in DAMAGE_TYPES:
        made.append(pairs_for_type(coords, dtype)); print("wrote", made[-1])
    made.append(tear_geometry_panel(coords)); print("wrote", made[-1])
    art, res = emit_example_artifact(coords)
    print("wrote", art, "| GT displacement", res.displacement.shape,
          "| removed", len(res.removed_index))
    (OUT / "README.md").write_text(
        f"# Damage examples (visual review before the sweep)\n\n"
        f"Section `{SECTION}` ({len(coords)} spots). Each panel pair is "
        f"**reference | damaged**, side by side, with the damage mask overlaid "
        f"(grey=intact, red=removed, blue=displaced, orange=folded, "
        f"green=stretched). Rows = severity 0 (control)..5; three example seeds "
        f"per severity.\n\n"
        f"- `<type>_pairs.png` for tear / tissue_loss / fold / stretch\n"
        f"- `tear_geometries.png` - straight / curved / branching tear paths\n"
        f"- `example_artifact.npz` - emitted GT (mask + per-spot displacement)\n",
        encoding="utf-8")
    print("STOP GATE: rendered", len(made), "figures ->", OUT)


if __name__ == "__main__":
    main()
