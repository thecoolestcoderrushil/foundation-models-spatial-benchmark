"""
Scoring for a predicted registration.

All errors are in spot-pitch units (median nearest-neighbour distance of the
reference) so they are comparable across donors and sections.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def spot_pitch(coords):
    coords = np.asarray(coords, float)
    d, _ = cKDTree(coords).query(coords, k=2)
    return float(np.median(d[:, 1]))


def registration_error(pred_xy, gt_xy, have, pitch):
    """Mean & median per-spot displacement error over spots that have GT.
    Returns dict in pitch units plus the scored count."""
    pred = np.asarray(pred_xy, float)
    gt = np.asarray(gt_xy, float)
    err = np.linalg.norm(pred - gt, axis=1)
    valid = np.isfinite(err) & np.asarray(have, bool)
    e = err[valid]
    if e.size == 0:
        return dict(rmse_pitch=np.nan, mean_pitch=np.nan, median_pitch=np.nan,
                    p90_pitch=np.nan, n_scored=0)
    e = e / pitch
    return dict(mean_pitch=float(e.mean()), median_pitch=float(np.median(e)),
                p90_pitch=float(np.percentile(e, 90)),
                rmse_pitch=float(np.sqrt((e ** 2).mean())), n_scored=int(e.size))


def label_transfer_accuracy(pred_xy, ref, mov, mov_have, layer_ref, layer_mov):
    """Map each moving spot to its nearest reference spot in the PREDICTED frame,
    transfer that reference spot's cortical-layer label, and compare with the
    moving spot's own label. Masks NA on both sides. Higher = registration keeps
    biology aligned."""
    pred = np.asarray(pred_xy, float)
    ref_xy = np.asarray(ref.obsm["spatial"], float)
    layer_ref = np.asarray(layer_ref).astype(str)
    layer_mov = np.asarray(layer_mov).astype(str)
    ok = np.isfinite(pred).all(1) & np.asarray(mov_have, bool)
    if ok.sum() == 0:
        return dict(label_acc=np.nan, n_label=0)
    nn = cKDTree(ref_xy).query(pred[ok])[1]
    transferred = layer_ref[nn]
    truth = layer_mov[ok]
    valid = (truth != "NA") & (transferred != "NA") & (truth != "nan") & (transferred != "nan")
    if valid.sum() == 0:
        return dict(label_acc=np.nan, n_label=0)
    acc = float((transferred[valid] == truth[valid]).mean())
    return dict(label_acc=acc, n_label=int(valid.sum()))


def is_degenerate(pred_xy, ref, min_spread_frac=0.05):
    """A transform is degenerate if the prediction collapses (near-constant) or is
    largely non-finite - a silent failure mode distinct from a crash."""
    pred = np.asarray(pred_xy, float)
    finite = np.isfinite(pred).all(1)
    if finite.mean() < 0.5:
        return True, "majority non-finite"
    p = pred[finite]
    ref_xy = np.asarray(ref.obsm["spatial"], float)
    ref_span = np.linalg.norm(ref_xy.max(0) - ref_xy.min(0))
    pred_span = np.linalg.norm(p.max(0) - p.min(0))
    if pred_span < min_spread_frac * ref_span:
        return True, f"collapsed (span {pred_span:.0f} << ref {ref_span:.0f})"
    return False, ""
