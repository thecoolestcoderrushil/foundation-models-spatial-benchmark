"""
DLPFC serial-section corpus (spatialLIBD, 12 sections, 3 donors).

This is the *entire* corpus by design: adjacent sections within a donor are true
serial pairs, and the Visium array coordinate (array_row, array_col) gives a
spot-level correspondence between them that is invariant to any coordinate-space
damage we apply - the registration ground truth.

We read the 12 sections from the local ARCA data cache (gitignored, large). No
other datasets are added.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATA = Path(r"C:/Users/karti/arca/data")   # local spatialLIBD cache (h5ad, gitignored)

# 3 donors x 4 consecutive sections (spatialLIBD sample ids).
DONORS = {
    "Br5292": ["151507", "151508", "151509", "151510"],
    "Br5595": ["151669", "151670", "151671", "151672"],
    "Br8100": ["151673", "151674", "151675", "151676"],
}
LAYER_KEY = "layer"          # 7 cortical layers: Layer1..6 + WM (a few NA)


def serial_pairs():
    """Yield (donor, ref_id, mov_id) for every within-donor adjacent pair."""
    for donor, sections in DONORS.items():
        for i in range(len(sections) - 1):
            yield donor, sections[i], sections[i + 1]


def load_section(sample_id):
    import anndata as ad
    a = ad.read_h5ad(DATA / f"DLPFC_{sample_id}.h5ad")
    a.obsm["spatial"] = np.asarray(a.obsm["spatial"], float)
    a.obs["array_row"] = a.obs["array_row"].astype(int)
    a.obs["array_col"] = a.obs["array_col"].astype(int)
    return a


def array_bridge(ref, mov):
    """Ground-truth correspondence: for each mov spot, the ref coordinate at the
    SAME Visium array position. Returns (gt_xy[n_mov,2], have[n_mov] bool)."""
    key = {(int(r), int(c)): i for i, (r, c) in
           enumerate(zip(ref.obs["array_row"], ref.obs["array_col"]))}
    ref_xy = np.asarray(ref.obsm["spatial"], float)
    gt = np.full((mov.n_obs, 2), np.nan)
    have = np.zeros(mov.n_obs, bool)
    for j, (r, c) in enumerate(zip(mov.obs["array_row"], mov.obs["array_col"])):
        i = key.get((int(r), int(c)))
        if i is not None:
            gt[j] = ref_xy[i]
            have[j] = True
    return gt, have


def common_genes(a, b):
    g = np.intersect1d(np.asarray(a.var_names), np.asarray(b.var_names))
    return g


def layer_labels(adata):
    if LAYER_KEY in adata.obs:
        return adata.obs[LAYER_KEY].astype(str).to_numpy()
    return np.array(["NA"] * adata.n_obs)
