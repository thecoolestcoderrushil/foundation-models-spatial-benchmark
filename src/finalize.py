"""
Turn results/benchmark_results.csv into FINDINGS.md (+ a machine-readable
leaderboard) with REAL numbers only. Run after the sweep (or on partial data;
it states how many cells it saw). Results are computed first; the interpretation
prose is derived from the numbers, not assumed.

  python src/finalize.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "benchmark_results.csv"
FINDINGS = ROOT / "FINDINGS.md"

PRIMARY = ["tear", "tissue_loss", "fold", "stretch"]
SECONDARY = ["tear_edge"]
BREAK_PITCH = 5.0        # "broken" once median error exceeds this many spot-pitches


def load():
    rows = []
    with open(CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def agg_median(rows):
    """(method,dtype,sev) -> mean of per-cell median_pitch over pairs/seeds,
    excluding failed cells. Also failure rate per (method,dtype,sev)."""
    med = defaultdict(list)
    fail = defaultdict(lambda: [0, 0])
    for r in rows:
        k = (r["method"], r["damage_type"], int(r["severity"]))
        if r["metric"] == "failed":
            fail[k][1] += 1
            fail[k][0] += int(num(r["value"]) or 0)
        if r["metric"] == "median_pitch" and int(num(r["failed"]) or 0) == 0:
            v = num(r["value"])
            if np.isfinite(v):
                med[k].append(v)
    med = {k: float(np.mean(v)) for k, v in med.items() if v}
    fr = {k: (f / t if t else np.nan) for k, (f, t) in fail.items()}
    return med, fr


def methods_in(rows):
    return sorted({r["method"] for r in rows})


def severities_in(rows, dtype):
    return sorted({int(r["severity"]) for r in rows if r["damage_type"] == dtype})


def breaks_at(med, fr, method, dtype, sevs):
    """First severity at which median error exceeds BREAK_PITCH or failure>50%."""
    for s in sevs:
        m = med.get((method, dtype, s))
        f = fr.get((method, dtype, s), 0)
        if (m is not None and m > BREAK_PITCH) or (f is not None and f > 0.5):
            return s
    return None


def main():
    if not CSV.exists():
        print("no benchmark_results.csv yet"); return
    rows = load()
    n_cells = sum(1 for r in rows if r["metric"] == "median_pitch")
    med, fr = agg_median(rows)
    methods = methods_in(rows)

    L = []
    L.append("# Findings - registration degradation under tissue damage\n")
    L.append(f"_Computed from `results/benchmark_results.csv` ({n_cells} "
             f"method-cells). Numbers are mean per-cell median spot-pitch error "
             f"over serial pairs and seeds, excluding failed cells._\n")

    # headline table: median error by method x severity, per primary damage type
    L.append("## Median registration error (spot-pitches) by severity\n")
    for dtype in PRIMARY + SECONDARY:
        sevs = severities_in(rows, dtype)
        if not sevs:
            continue
        L.append(f"\n### {dtype}\n")
        L.append("| method | " + " | ".join(f"sev {s}" for s in sevs) + " | breaks at |")
        L.append("|" + "---|" * (len(sevs) + 2))
        for m in methods:
            cells = []
            for s in sevs:
                v = med.get((m, dtype, s))
                f = fr.get((m, dtype, s), 0)
                if v is None:
                    cells.append("fail" if (f and f > 0.5) else "-")
                else:
                    cells.append(f"{v:.2f}" + ("!" if f and f > 0.5 else ""))
            b = breaks_at(med, fr, m, dtype, sevs)
            L.append(f"| `{m}` | " + " | ".join(cells) + f" | {b if b is not None else 'never'} |")

    # rigid-vs-OT at lowest damage (severity>0 min) - the "is the floor competitive" question
    L.append("\n## Is the rigid floor competitive?\n")
    lo = 2
    lines = []
    for dtype in PRIMARY:
        rg = med.get(("rigid", dtype, lo))
        p2 = med.get(("paste2", dtype, lo))
        if rg is not None and p2 is not None:
            verdict = ("rigid <= paste2" if rg <= p2 else "paste2 < rigid")
            lines.append(f"- **{dtype}** @ sev {lo}: rigid {rg:.2f} vs paste2 {p2:.2f} pitch -> {verdict}")
    L.extend(lines or ["- (insufficient data yet)"])

    # failure summary
    L.append("\n## Failures (crashes + degenerate/timeout)\n")
    L.append("| method | damage | severity | failure rate |")
    L.append("|---|---|---|---|")
    any_fail = False
    for (m, d, s), rate in sorted(fr.items()):
        if rate and rate > 0:
            any_fail = True
            L.append(f"| `{m}` | {d} | {s} | {rate*100:.0f}% |")
    if not any_fail:
        L.append("| - | - | - | none observed |")

    # honest interpretation derived from the numbers
    L.append("\n## Interpretation (from the numbers above)\n")
    notes = []
    # rigid competitiveness
    comp = [dtype for dtype in PRIMARY
            if med.get(("rigid", dtype, lo)) is not None
            and med.get(("paste2", dtype, lo)) is not None
            and med[("rigid", dtype, lo)] <= med[("paste2", dtype, lo)]]
    if comp:
        notes.append(f"- The parameter-free rigid ICP floor is at least as good as "
                     f"PASTE2 at low damage (sev {lo}) on: {', '.join(comp)}. "
                     f"Expensive OT does not buy accuracy there.")
    # who breaks first per type
    for dtype in PRIMARY:
        sevs = severities_in(rows, dtype)
        bs = {m: breaks_at(med, fr, m, dtype, sevs) for m in methods}
        bs = {m: b for m, b in bs.items() if b is not None}
        if bs:
            worst = min(bs, key=lambda m: bs[m])
            notes.append(f"- On **{dtype}**, `{worst}` breaks earliest "
                         f"(median error > {BREAK_PITCH:g} pitch by severity {bs[worst]}).")
    L.extend(notes or ["- (fill on more data)"])

    L.append("\n## Excluded / caveats\n")
    L.append("- **STalign**: real LDDMM API wired (py3.11 + numpy<2) but did not "
             "converge beating no-op on control transforms across tried "
             "hyperparameters; excluded rather than shown mis-tuned.\n"
             "- **PASTE**: POT>=0.9 `line_search` API break in its FGW; PASTE2 used.\n"
             "- **Damage realism**: primary tears are interior cracks (closed "
             "outline); the boundary-reaching `tear_edge` condition probes the gap.\n"
             "- Moving section = damaged self + known rigid misalignment (isolates "
             "damage; omits inter-section biology).")

    FINDINGS.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {FINDINGS} from {n_cells} method-cells")


if __name__ == "__main__":
    main()
