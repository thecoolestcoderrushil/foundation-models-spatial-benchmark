"""
The degradation sweep (harness only - DO NOT run until the damage renders are
approved).

For every serial pair, damage type, severity level, and seed, damage the moving
section with a known ground-truth transform, register it to the (undamaged)
reference with each available method, and score. One CSV row per
(donor x pair x damage-type x severity x seed x method x metric).

Robustness: single-instance lock; resumable (skips cells already in the CSV, so a
reaped/killed job just relaunches and continues); incremental append after every
cell; per-cell try/except; progress log. Detached-friendly:

    nohup python -u src/run_benchmark.py > results/benchmark.console.log 2>&1 &

Grid (revised): spot density is what the damage model acts on, so we do NOT
subsample hard - sections are kept at ~2000 spots (--n-spots) so small tears stay
meaningful and label-transfer is not noise. We trim the grid elsewhere: 4
severities (--severities 0,2,4,5) and 5 seeds (--seeds). If a full run is still
too slow, cut section PAIRS before cutting spots (--max-pairs / --pairs-per-donor)
- never drop below ~2000 spots.
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

from damage import apply_damage, DAMAGE_TYPES, N_LEVELS   # noqa: E402
import datasets as D                                       # noqa: E402
import metrics as M                                        # noqa: E402
from methods import ALL_METHODS                            # noqa: E402

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


# ---- lock ---------------------------------------------------------------- #
def acquire_lock():
    try:
        fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        atexit.register(lambda: LOCK.exists() and LOCK.unlink())
        return True
    except FileExistsError:
        log(f"another instance holds {LOCK}; exiting.")
        return False


# ---- resume -------------------------------------------------------------- #
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


def subsample(ref, mov, n, seed):
    """Array-consistent subsample: keep the same array positions in both sections
    where possible so the array-bridge GT survives."""
    rng = np.random.default_rng(seed)
    def pick(a, n):
        if a.n_obs <= n:
            return np.arange(a.n_obs)
        return np.sort(rng.choice(a.n_obs, n, replace=False))
    return pick(ref, n), pick(mov, n)


def cell_rows(base, method, out, err, lab, degen_reason, n_spots):
    failed = bool(out.get("failed") or degen_reason)
    reason = out.get("reason", "") or degen_reason
    vals = dict(mean_pitch=err["mean_pitch"], median_pitch=err["median_pitch"],
                p90_pitch=err["p90_pitch"], label_acc=lab["label_acc"],
                runtime_s=out["runtime_s"], failed=int(failed),
                n_scored=err["n_scored"])
    rows = []
    for k in METRIC_KEYS:
        rows.append({**base, "method": method, "metric": k, "value": vals[k],
                     "n_spots": n_spots, "runtime_s": round(out["runtime_s"], 3),
                     "failed": int(failed), "reason": reason[:120], "timestamp": now()})
    return rows, failed


def run(args):
    methods = []
    for cls in ALL_METHODS:
        m = cls()
        ok, detail = m.available()
        if ok and m.name in args.methods:
            methods.append(m)
            log(f"method {m.name}: available")
        elif m.name in args.methods:
            log(f"method {m.name}: ABSENT - {detail} (skipped)")
    if not methods:
        log("no methods available; abort.")
        return

    seen = done_cells()
    log(f"resume: {len(seen)} cells already done")
    severities = args.severities
    total = 0

    pairs = list(D.serial_pairs())
    if args.pairs_per_donor:
        by_donor = {}
        kept = []
        for pr in pairs:
            by_donor.setdefault(pr[0], 0)
            if by_donor[pr[0]] < args.pairs_per_donor:
                kept.append(pr); by_donor[pr[0]] += 1
        pairs = kept
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    log(f"pairs: {len(pairs)} | severities={severities} | seeds={args.seeds} "
        f"| n_spots={args.n_spots}")

    for donor, ref_id, mov_id in pairs:
        try:
            ref = D.load_section(ref_id)
            mov0 = D.load_section(mov_id)
        except Exception as e:
            log(f"[{donor} {ref_id}->{mov_id}] load failed: {e}")
            continue
        ri, mi = subsample(ref, mov0, args.n_spots, seed=0)
        ref = ref[ri].copy(); mov0 = mov0[mi].copy()
        pitch = M.spot_pitch(np.asarray(ref.obsm["spatial"], float))
        gt_xy, have = D.array_bridge(ref, mov0)
        layer_ref = D.layer_labels(ref)
        layer_mov0 = D.layer_labels(mov0)
        mov_xy = np.asarray(mov0.obsm["spatial"], float)
        log(f"pair {donor} {ref_id}->{mov_id}: ref={ref.n_obs} mov={mov0.n_obs} "
            f"pitch={pitch:.0f} bridge={have.mean()*100:.0f}%")

        for dtype in DAMAGE_TYPES:
            for sev in severities:
                seeds = [0] if sev == 0 else list(range(args.seeds))
                for seed in seeds:
                    dmg = apply_damage(mov_xy, dtype, sev, seed=seed)
                    surv = dmg.survivor_index
                    mov = mov0[surv].copy()
                    mov.obsm["spatial"] = dmg.coords
                    gt_s, have_s = gt_xy[surv], have[surv]
                    layer_mov = layer_mov0[surv]
                    base = dict(donor=donor, ref=ref_id, mov=mov_id,
                                damage_type=dtype, severity=sev, seed=seed)
                    for m in methods:
                        key = (donor, mov_id, dtype, str(sev), str(seed), m.name)
                        if key in seen:
                            continue
                        try:
                            out = m.register(ref, mov)
                            err = M.registration_error(out["pred_xy"], gt_s, have_s, pitch)
                            lab = M.label_transfer_accuracy(
                                out["pred_xy"], ref, mov, have_s, layer_ref, layer_mov)
                            degen, dreason = M.is_degenerate(out["pred_xy"], ref)
                            rows, failed = cell_rows(base, m.name, out, err, lab,
                                                     dreason if degen else "", mov.n_obs)
                        except Exception as e:
                            out = dict(runtime_s=0.0)
                            rows = [{**base, "method": m.name, "metric": k, "value": "",
                                     "n_spots": mov.n_obs, "runtime_s": 0,
                                     "failed": 1, "reason": f"{type(e).__name__}: {e}"[:120],
                                     "timestamp": now()} for k in METRIC_KEYS]
                            failed = True
                            log(f"  CRASH {m.name} {dtype} s{sev} seed{seed}: {e}")
                        append_rows(rows)
                        total += 1
                    seen.add((donor, mov_id, dtype, str(sev), str(0 if sev == 0 else seed), m.name))
                log(f"  {donor} {dtype} sev{sev}: cells so far {total}")
    log(f"SWEEP COMPLETE. wrote ~{total} method-cells to {CSV}")
    log("DONE")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default="rigid,paste,paste2,stalign,gpsa",
                   help="comma list; absent ones auto-skip")
    p.add_argument("--n-spots", type=int, default=2000,
                   help="spots kept per section (do NOT drop below ~2000)")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--severities", default="0,2,4,5",
                   help="comma list of severity levels to sweep")
    p.add_argument("--pairs-per-donor", type=int, default=0,
                   help="cap adjacent pairs per donor (0 = all 3); cut pairs "
                        "before spots if too slow")
    p.add_argument("--max-pairs", type=int, default=0, help="global cap on pairs (0=all)")
    args = p.parse_args()
    args.methods = set(s.strip() for s in args.methods.split(","))
    args.severities = [int(x) for x in args.severities.split(",")]
    RESULTS.mkdir(exist_ok=True)
    if not acquire_lock():
        return
    log("=" * 70)
    log(f"DEGRADATION SWEEP start | methods={sorted(args.methods)} "
        f"n_spots={args.n_spots} seeds={args.seeds}")
    try:
        run(args)
    except Exception as e:
        log(f"TOP-LEVEL FAILURE: {e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
