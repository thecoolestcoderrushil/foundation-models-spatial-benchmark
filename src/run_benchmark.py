"""
The degradation sweep.

For every serial pair, damage type, severity level, and seed, we damage the
moving section with a known transform, apply a per-seed global rigid misalignment
(so registration is a real alignment task, not just local-damage cleanup), and
register to the reference with each available method. One CSV row per
(donor x pair x damage-type x severity x seed x method x metric).

Design decisions:
  * ~2000 spots/section (spot density is what the damage acts on; not subsampled
    to 600). Trim the grid via severities/seeds and cut PAIRS before spots.
  * Features: one shared HVG+PCA basis fit on the pooled pair, so OT (PASTE/PASTE2)
    is CPU-tractable (~70 s/cell at 2000 spots vs ~1 h with glmpca) and the
    cross-section features are comparable.
  * Global misalignment per seed: rotation +/-20 deg, translation +/-5 pitches,
    constant across severities for a seed so the degradation isolates damage.
  * Per-cell 10-min timeout via a persistent restart-on-timeout worker.
  * Secondary condition: boundary-reaching tear (tear_edge) at severity {0,4}.

Robustness: single-instance lock; resumable (skips done cells); incremental
append; per-cell try/except + timeout; progress log. Detached:

    nohup python -u src/run_benchmark.py > results/benchmark.console.log 2>&1 &
"""
from __future__ import annotations

import argparse
import atexit
import csv
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "damage"))
sys.path.insert(0, str(ROOT / "src"))

from damage import apply_damage                              # noqa: E402
import datasets as D                                         # noqa: E402
import metrics as M                                          # noqa: E402
from methods import ALL_METHODS, RigidAffine                 # noqa: E402
from _worker import TimeoutWorker                             # noqa: E402

RESULTS = ROOT / "results"
CSV = RESULTS / "benchmark_results.csv"
LOG = RESULTS / "benchmark.log"
LOCK = RESULTS / "benchmark.lock"

FIELDS = ["donor", "ref", "mov", "damage_type", "severity", "seed", "method",
          "metric", "value", "n_spots", "runtime_s", "failed", "reason", "timestamp"]
METRIC_KEYS = ["mean_pitch", "median_pitch", "p90_pitch", "label_acc",
               "runtime_s", "failed", "n_scored"]


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    RESULTS.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="ascii", errors="replace") as fh:
        fh.write(line + "\n")


def acquire_lock():
    try:
        fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        atexit.register(lambda: LOCK.exists() and LOCK.unlink())
        return True
    except FileExistsError:
        log(f"another instance holds {LOCK}; exiting.")
        return False


def done_cells():
    if not CSV.exists():
        return set()
    seen = set()
    with open(CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            seen.add((r["donor"], r["mov"], r["damage_type"], r["severity"],
                      r["seed"], r["method"]))
    return seen


def append_rows(rows):
    exists = CSV.exists()
    with open(CSV, "a", newline="", encoding="ascii", errors="replace") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# per-pair features + per-cell geometry
# --------------------------------------------------------------------------- #
def shared_features(ref, mov, hvg=2000, k=30):
    """One shared HVG+PCA basis on the pooled pair -> comparable, low-dim
    features (obsm X_pca) that make OT tractable at 2000 spots."""
    import scanpy as sc
    import anndata as ad

    def prep(a):
        a = a.copy()
        sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
        return a
    A, B = prep(ref), prep(mov)
    g = np.intersect1d(np.asarray(A.var_names), np.asarray(B.var_names))
    A, B = A[:, g].copy(), B[:, g].copy()
    sc.pp.highly_variable_genes(A, n_top_genes=int(min(hvg, A.n_vars)))
    hv = A.var_names[A.var["highly_variable"]]
    pooled = ad.concat([A[:, hv], B[:, hv]])
    sc.pp.pca(pooled, n_comps=int(min(k, pooled.n_vars - 1)))
    P = pooled.obsm["X_pca"]
    return P[:A.n_obs].astype(np.float32), P[A.n_obs:].astype(np.float32)


def rigid_misalign(coords, seed, pitch, max_rot_deg=10.0, max_trans_pitch=3.0):
    rng = np.random.default_rng((0xA11, int(seed)))
    th = np.deg2rad(rng.uniform(-max_rot_deg, max_rot_deg))
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    t = rng.uniform(-max_trans_pitch, max_trans_pitch, 2) * pitch
    cen = coords.mean(0)
    return (coords - cen) @ R.T + cen + t


def slim(src, spatial, xpca):
    import anndata as ad
    cols = [c for c in ("array_row", "array_col", "layer") if c in src.obs]
    s = ad.AnnData(X=np.zeros((spatial.shape[0], 1), np.float32),
                   obs=src.obs[cols].copy())
    s.obsm["spatial"] = np.asarray(spatial, float)
    s.obsm["X_pca"] = np.asarray(xpca, np.float32)
    return s


def cell_rows(base, method, out, err, lab, degen_reason, n_spots):
    failed = bool(out.get("failed") or degen_reason)
    reason = (out.get("reason", "") or degen_reason)
    vals = dict(mean_pitch=err["mean_pitch"], median_pitch=err["median_pitch"],
                p90_pitch=err["p90_pitch"], label_acc=lab["label_acc"],
                runtime_s=out["runtime_s"], failed=int(failed),
                n_scored=err["n_scored"])
    rows = [{**base, "method": method, "metric": k, "value": vals[k],
             "n_spots": n_spots, "runtime_s": round(out["runtime_s"], 2),
             "failed": int(failed), "reason": str(reason)[:120], "timestamp": now()}
            for k in METRIC_KEYS]
    return rows, failed


def run(args):
    avail = []
    for cls in ALL_METHODS:
        m = cls()
        ok, detail = m.available()
        if m.name not in args.methods:
            continue
        if ok:
            avail.append(m); log(f"method {m.name}: available - {detail}")
        else:
            log(f"method {m.name}: ABSENT - {detail} (skipped)")
    if not avail:
        log("no methods available; abort."); return

    worker = TimeoutWorker()
    seen = done_cells()
    log(f"resume: {len(seen)} cells already done")

    pairs = list(D.serial_pairs())
    if args.pairs_per_donor:
        by, kept = {}, []
        for pr in pairs:
            by.setdefault(pr[0], 0)
            if by[pr[0]] < args.pairs_per_donor:
                kept.append(pr); by[pr[0]] += 1
        pairs = kept
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]

    # primary types at the requested severities + secondary boundary tear at {0,4}
    type_sevs = [(t, args.severities) for t in ("tear", "tissue_loss", "fold", "stretch")]
    type_sevs.append(("tear_edge", [s for s in (0, 4) if s in args.severities or s == 0] or [0, 4]))
    log(f"pairs={len(pairs)} methods={[m.name for m in avail]} "
        f"severities={args.severities} seeds={args.seeds} n_spots={args.n_spots} "
        f"timeout={args.timeout}s")

    total = 0
    for donor, ref_id, mov_id in pairs:
        try:
            ref0 = D.load_section(ref_id); mov0 = D.load_section(mov_id)
        except Exception as e:
            log(f"[{donor} {ref_id}->{mov_id}] load failed: {e}"); continue
        rng = np.random.default_rng(0)
        ri = np.sort(rng.choice(ref0.n_obs, min(args.n_spots, ref0.n_obs), replace=False))
        mi = np.sort(rng.choice(mov0.n_obs, min(args.n_spots, mov0.n_obs), replace=False))
        ref0, mov0 = ref0[ri].copy(), mov0[mi].copy()
        pitch = M.spot_pitch(np.asarray(ref0.obsm["spatial"], float))
        gt, have = D.array_bridge(ref0, mov0)
        try:
            Xr, Xm = shared_features(ref0, mov0)
        except Exception as e:
            log(f"[{donor}] feature build failed: {e}"); continue
        layer_ref = D.layer_labels(ref0); layer_mov0 = D.layer_labels(mov0)
        ref_slim = slim(ref0, np.asarray(ref0.obsm["spatial"], float), Xr)
        mov_xy = np.asarray(mov0.obsm["spatial"], float)
        log(f"pair {donor} {ref_id}->{mov_id}: n={ref0.n_obs}/{mov0.n_obs} "
            f"pitch={pitch:.0f} bridge={have.mean()*100:.0f}%")

        for dtype, sevs in type_sevs:
            for sev in sevs:
                seeds = [0] if sev == 0 else list(range(args.seeds))
                for seed in seeds:
                    dmg = apply_damage(mov_xy, dtype, sev, seed=seed)
                    surv = dmg.survivor_index
                    coords_final = rigid_misalign(dmg.coords, seed, pitch)
                    mov_slim = slim(mov0[surv], coords_final, Xm[surv])
                    gt_s, have_s = gt[surv], have[surv]
                    layer_mov = layer_mov0[surv]
                    base = dict(donor=donor, ref=ref_id, mov=mov_id,
                                damage_type=dtype, severity=sev, seed=seed)
                    for m in avail:
                        key = (donor, mov_id, dtype, str(sev), str(seed), m.name)
                        if key in seen:
                            continue
                        try:
                            if m.name == "rigid":
                                out = m.register(ref_slim, mov_slim)   # fast, in-proc
                            else:
                                out, _ = worker.call(m.name, ref_slim, mov_slim, args.timeout)
                            pred = out.get("pred_xy")
                            if pred is None:
                                err = dict(mean_pitch="", median_pitch="", p90_pitch="", n_scored=0)
                                lab = dict(label_acc="", n_label=0); dreason = "no prediction"
                            else:
                                err = M.registration_error(pred, gt_s, have_s, pitch)
                                lab = M.label_transfer_accuracy(pred, ref_slim, mov_slim,
                                                                have_s, layer_ref, layer_mov)
                                degen, dreason = M.is_degenerate(pred, ref_slim)
                                dreason = dreason if degen else ""
                            rows, failed = cell_rows(base, m.name, out, err, lab, dreason,
                                                     mov_slim.n_obs)
                        except Exception as e:
                            rows = [{**base, "method": m.name, "metric": k, "value": "",
                                     "n_spots": mov_slim.n_obs, "runtime_s": 0, "failed": 1,
                                     "reason": f"{type(e).__name__}: {e}"[:120],
                                     "timestamp": now()} for k in METRIC_KEYS]
                            failed = True
                            log(f"  CRASH {m.name} {dtype} s{sev} seed{seed}: {e}")
                        append_rows(rows); total += 1
                        seen.add(key)
                        rt = rows[0]["runtime_s"]
                        log(f"  {m.name:7s} {dtype:11s} sev{sev} seed{seed}: "
                            f"med={rows[1]['value']} rt={rt}s failed={rows[0]['failed']}")
    worker.close()
    log(f"SWEEP COMPLETE ({total} method-cells this run) -> {CSV}")
    log("DONE")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default="rigid,paste2,gpsa",
                   help="comma list; absent auto-skip. PASTE excluded (POT>=0.9 "
                        "line_search API break); STalign experimental/off by default")
    p.add_argument("--n-spots", type=int, default=2000)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--severities", default="0,2,4,5")
    p.add_argument("--pairs-per-donor", type=int, default=0)
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--timeout", type=int, default=600, help="per-cell wall-clock budget (s)")
    args = p.parse_args()
    args.methods = set(s.strip() for s in args.methods.split(","))
    args.severities = [int(x) for x in args.severities.split(",")]
    RESULTS.mkdir(exist_ok=True)
    if not acquire_lock():
        return
    log("=" * 70)
    log(f"DEGRADATION SWEEP start | methods={sorted(args.methods)}")
    try:
        run(args)
    except Exception as e:
        log(f"TOP-LEVEL FAILURE: {e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
