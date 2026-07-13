"""
Figures from results/benchmark_results.csv (long format).

  degradation_curves.png  median registration error vs severity, one line per
                          method, faceted by damage type (the leaderboard view).
  failure_rates.png       fraction of degenerate/crashed runs per method x type.
  runtime_scatter.png     runtime vs damage severity per method.

Run after the sweep:  python src/plots.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CSV = RESULTS / "benchmark_results.csv"

METHOD_COLORS = {"rigid": "#9aa0a6", "paste": "#3d5afe", "paste2": "#6633ee",
                 "stalign": "#d1495b", "gpsa": "#12a150"}
DAMAGE_TYPES = ["tear", "tissue_loss", "fold", "stretch"]


def load():
    rows = []
    with open(CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def curve_data(rows, metric="median_pitch"):
    """(method, dtype, severity) -> mean value over pairs/seeds (excl failures)."""
    agg = defaultdict(list)
    for r in rows:
        if r["metric"] != metric or int(_num(r["failed"]) or 0) == 1:
            continue
        v = _num(r["value"])
        if np.isfinite(v):
            agg[(r["method"], r["damage_type"], int(r["severity"]))].append(v)
    return {k: float(np.mean(v)) for k, v in agg.items() if v}


def degradation_curves(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = curve_data(rows, "median_pitch")
    methods = sorted({m for (m, _, _) in data})
    fig, axes = plt.subplots(1, len(DAMAGE_TYPES), figsize=(4.4 * len(DAMAGE_TYPES), 4),
                             sharey=True)
    for ax, dt in zip(axes, DAMAGE_TYPES):
        for m in methods:
            xs, ys = [], []
            for s in range(6):
                if (m, dt, s) in data:
                    xs.append(s); ys.append(data[(m, dt, s)])
            if xs:
                ax.plot(xs, ys, "-o", ms=4, color=METHOD_COLORS.get(m, "#333"), label=m)
        ax.set_title(dt); ax.set_xlabel("severity"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("median registration error (spot-pitches)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Registration degradation vs tissue-damage severity", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS / "degradation_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def failure_rates(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fr = defaultdict(lambda: [0, 0])   # (method,dtype) -> [failed, total]
    for r in rows:
        if r["metric"] != "failed":
            continue
        k = (r["method"], r["damage_type"])
        fr[k][1] += 1
        fr[k][0] += int(_num(r["value"]) or 0)
    methods = sorted({m for (m, _) in fr})
    x = np.arange(len(DAMAGE_TYPES)); w = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, m in enumerate(methods):
        ys = [(fr[(m, dt)][0] / fr[(m, dt)][1]) if fr[(m, dt)][1] else 0
              for dt in DAMAGE_TYPES]
        ax.bar(x + i * w, ys, w, color=METHOD_COLORS.get(m, "#333"), label=m)
    ax.set_xticks(x + w * (len(methods) - 1) / 2); ax.set_xticklabels(DAMAGE_TYPES)
    ax.set_ylabel("failure rate (crashes + degenerate)"); ax.legend(frameon=False)
    ax.set_title("Failure rate by method and damage type")
    fig.tight_layout()
    fig.savefig(RESULTS / "failure_rates.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def runtime_scatter(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["metric"] != "runtime_s":
            continue
        pts[r["method"]][int(r["severity"])].append(_num(r["value"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, bysev in pts.items():
        xs = sorted(bysev); ys = [np.nanmean(bysev[s]) for s in xs]
        ax.plot(xs, ys, "-o", color=METHOD_COLORS.get(m, "#333"), label=m)
    ax.set_xlabel("severity"); ax.set_ylabel("mean runtime per registration (s)")
    ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend(frameon=False)
    ax.set_title("Compute cost vs damage severity")
    fig.tight_layout()
    fig.savefig(RESULTS / "runtime_scatter.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    if not CSV.exists():
        print(f"no {CSV} yet - run the sweep first (src/run_benchmark.py)")
        return
    rows = load()
    degradation_curves(rows)
    failure_rates(rows)
    runtime_scatter(rows)
    print("wrote degradation_curves.png, failure_rates.png, runtime_scatter.png")


if __name__ == "__main__":
    main()
