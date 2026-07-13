# How do spatial registration methods degrade as tissue damage increases?

Serial-section spatial transcriptomics needs registration: align section *i* to
section *i+1*. Real sections are **damaged** — torn on the slide, folded, cut,
regions lost. Registration methods are benchmarked almost exclusively on clean
tissue. **Nobody has systematically characterized where they break as damage
increases.** This repo does, and publishes three things: a reproducible **damage
generator**, a **harness** that sweeps it against every registration method, and
a **leaderboard** of degradation curves.

CPU-only. Public data only (spatialLIBD DLPFC). No foundation models, nothing
gated, no downloaded weights.

## The corpus (fixed)

spatialLIBD DLPFC, all 12 sections, 3 donors × 4 consecutive sections. Adjacent
sections within a donor are true serial pairs; the Visium array coordinate
`(array_row, array_col)` gives a **spot-level ground-truth correspondence** that
is invariant to any coordinate-space damage we apply. That is the entire corpus —
no other datasets are added.

## The damage model (the core contribution) — `damage/`

A standalone, pip-installable, seeded generator. Four physically-motivated damage
types, each with an integer severity level `0..5` (0 = undamaged control) that
maps to an interpretable physical parameter, and an **exact per-spot ground-truth
displacement** so registration error is measurable to the spot.

| type | mechanism | severity parameter |
|---|---|---|
| **tear** | a cut propagating along a path (straight / curved / branching); the two lips separate, the blade path is removed | gap opened, in spot-pitches |
| **tissue_loss** | a contiguous edge-anchored region removed | fraction of spots lost (5–50%) |
| **fold** | a flap doubled over a fold line; folded spots overlap the body | flap fraction of section extent |
| **stretch** | local non-rigid radial deformation near a boundary | peak displacement, in spot-pitches |

```python
from damage import apply_damage
res = apply_damage(coords, "tear", severity=3, seed=0)   # deterministic
res.coords          # damaged coordinates of surviving spots
res.displacement    # exact ground-truth forward transform (invert to register)
res.mask            # per-spot: intact / removed / displaced / folded / stretched
```

Every `(damage_type, severity, seed)` reproduces the same section bit-for-bit.
Damage renders for visual review: **`results/damage_examples/`**.

## Methods (installed honestly)

`rigid` (parameter-free similarity floor, in-repo) · **PASTE** · **PASTE2**
(partial OT, built for tissue loss) · **STalign** (LDDMM) · **GPSA**
(Gaussian-process alignment). Each is probed for real; unavailable methods are
recorded with their exact error in `results/methods_env.md` and skipped — never
stubbed with a fabricated API. This environment (Windows, CPU, Python 3.14):
`rigid`, `PASTE`, `PASTE2` available; `STalign` fails to build; `GPSA` needs a
local install. A prior foundation-model install audit is in
`results/env_check.md` (all five single-cell FMs unusable here — motivating the
CPU-only, classical-methods scope).

## Metrics

Registration error vs GT (mean + median spot displacement, in pitches),
cortical-layer label-transfer accuracy, runtime, and failure rate (crashes +
degenerate/collapsed transforms).

## Reproduce

```bash
pip install -r requirements.txt
pip install -e damage                      # the generator
python src/render_damage_examples.py       # STOP-GATE: renders for review
python src/methods.py                       # methods availability -> results/methods_env.md
# after reviewing renders:
nohup python -u src/run_benchmark.py > results/benchmark.console.log 2>&1 &
python src/plots.py                         # figures from results/benchmark_results.csv
```

The sweep is resumable (skips completed cells), incremental (appends per cell),
single-instance-locked, and detached-friendly.

## Status

**Stop gate: the damage model is implemented and rendered; the sweep has not been
run.** Review `results/damage_examples/` first.

## Authors

Rushil Maniar (Sutura Genomics), Sean Lee (Sutura Genomics), Sunyoung Lee, MD
(MD Anderson Cancer Center). The manuscript source is in `paper/`.
