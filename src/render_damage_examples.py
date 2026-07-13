"""
STOP-GATE deliverable: render damage examples for visual review BEFORE any sweep.

For a representative DLPFC section we render, for every damage type, a grid of
severity (0..5) x seed (3 examples) so the damage model can be eyeballed. We also
render a dedicated tear-geometry panel (straight / curved / branching) and emit
one example's ground-truth artifact (mask + displacement) so the emitted-artifact
contract is demonstrable.

Outputs -> results/damage_examples/
  <type>_grid.png            severity x seed grid, coloured by damage label
  tear_geometries.png        straight / curved / branching side by side
  example_artifact.npz       mask + displacement + survivor/removed indices for one case
  README.md                  what each panel shows

Usage:
  python src/render_damage_examples.py
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
import anndata as ad

from damage import apply_damage, DAMAGE_TYPES, SEVERITY, TEAR_PATHS
from damage.render import plot_damage

DATA = Path(r"C:/Users/karti/arca/data")
SECTION = "DLPFC_151507"
OUT = ROOT / "results" / "damage_examples"
SEEDS = (0, 1, 2)
LEVELS = list(range(0, 6))          # 0 control .. 5 max


def load_section():
    A = ad.read_h5ad(DATA / f"{SECTION}.h5ad")
    return np.asarray(A.obsm["spatial"], float)


def severity_label(dtype, lvl):
    key, vals = next(iter(SEVERITY[dtype].items()))
    unit = {"gap_pitch": "gap px-pitch", "area_frac": "area lost",
            "fold_flap": "flap", "flap_frac": "flap frac", "peak_pitch": "peak pitch"}.get(key, key)
    v = vals[lvl]
    if key == "area_frac":
        return f"sev {lvl}\n{int(v*100)}% lost"
    if key == "flap_frac":
        return f"sev {lvl}\nflap {int(v*100)}%"
    return f"sev {lvl}\n{key.split('_')[0]} {v:g}"


def grid_for_type(coords, dtype):
    nrow, ncol = len(SEEDS), len(LEVELS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.05 * ncol, 2.05 * nrow))
    for r, seed in enumerate(SEEDS):
        for c, lvl in enumerate(LEVELS):
            ax = axes[r, c]
            res = apply_damage(coords, dtype, lvl, seed=seed)
            plot_damage(res, ax=ax, point_size=2.5)
            if r == 0:
                ax.set_title(severity_label(dtype, lvl), fontsize=8)
            if c == 0:
                ax.set_ylabel(f"seed {seed}", fontsize=8)
    fig.suptitle(f"Damage type: {dtype}   (section {SECTION}, {len(coords)} spots)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    p = OUT / f"{dtype}_grid.png"
    fig.savefig(p, dpi=115, bbox_inches="tight")
    plt.close(fig)
    return p


def tear_geometry_panel(coords):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    for ax, kind in zip(axes, TEAR_PATHS):
        res = apply_damage(coords, "tear", severity=4, seed=0, tear_path=kind)
        plot_damage(res, ax=ax, point_size=2.5, title=f"tear path: {kind}")
    fig.suptitle("Tear geometries at severity 4 (straight / curved / branching)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    p = OUT / "tear_geometries.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p


def emit_example_artifact(coords):
    res = apply_damage(coords, "tear", severity=3, seed=0, tear_path="curved")
    p = OUT / "example_artifact.npz"
    np.savez_compressed(
        p,
        input_coords=coords,
        damaged_coords=res.coords,
        original_coords=res.original_coords,
        gt_displacement=res.displacement,
        survivor_index=res.survivor_index,
        removed_index=res.removed_index,
        mask=res.mask.astype("U12"),
        meta=np.array(str(res.meta), dtype=object),
    )
    return p, res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    coords = load_section()
    made = []
    for dtype in DAMAGE_TYPES:
        made.append(grid_for_type(coords, dtype))
        print("wrote", made[-1])
    made.append(tear_geometry_panel(coords))
    print("wrote", made[-1])
    art, res = emit_example_artifact(coords)
    print("wrote", art, "| GT displacement shape", res.displacement.shape,
          "| removed", len(res.removed_index))

    (OUT / "README.md").write_text(
        "# Damage examples (visual review before the sweep)\n\n"
        f"Section: `{SECTION}` ({len(coords)} spots). Severity 0 = undamaged "
        "control; 1..5 increasing. Colours: grey=intact, red=removed, "
        "blue=displaced (tear), orange=folded, green=stretched.\n\n"
        "- `tear_grid.png`, `tissue_loss_grid.png`, `fold_grid.png`, "
        "`stretch_grid.png` - severity (cols) x seed (rows).\n"
        "- `tear_geometries.png` - straight / curved / branching tear paths.\n"
        "- `example_artifact.npz` - the emitted ground-truth artifact for one "
        "damaged section (mask + per-spot GT displacement + survivor/removed "
        "indices).\n", encoding="utf-8")
    print("STOP GATE: rendered", len(made), "figures ->", OUT)


if __name__ == "__main__":
    main()
