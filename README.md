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
stubbed with a fabricated API.

**Environment.** All five methods run on **Python 3.11** (CPU). STalign does *not*
build on Python 3.14 (its pinned NumPy fails to compile) but installs and imports
cleanly on 3.11, so the benchmark environment is 3.11. GPSA is installed locally
from `baselines/GPSA`. The sweep is therefore run with the 3.11 interpreter:

```bash
uv venv --python 3.11 .venv311
uv pip install --python .venv311 -r requirements.txt \
    "git+https://github.com/raphael-group/paste2.git" \
    "git+https://github.com/JEFworks-Lab/STalign.git"
uv pip install --python .venv311 --no-deps -e baselines/GPSA   # torch installed separately
```

A prior foundation-model install audit is in `results/env_check.md` (all five
single-cell FMs unusable on this host — motivating the classical-methods scope).

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

## Limitations

- **The tear is a crack in a fixed plate, not a boundary-reaching separation.**
  For the four primary damage types the section's outer boundary stays a closed
  polygon while damage opens *internally*. Real torn sections have lips that
  separate at the tissue edge, opening the outline itself. Our primary tears do
  not, which makes the task **easier than reality**: a method can anchor on the
  intact outline. We keep it this way because exact per-spot ground truth is
  worth more here than visual realism. A secondary **`tear_edge`** condition
  (boundary-reaching tear whose slit opens the outline) is included at severity
  {0,4} to probe this gap, but the primary leaderboard is on interior damage.
- **Sections are damaged copies of themselves plus a synthetic rigid
  misalignment**, not two biologically distinct serial sections; this isolates
  the damage effect but omits real inter-section biological variation.
- **STalign is excluded from the leaderboard.** It installs and imports only on
  Python 3.11 + numpy<2, and its adapter uses the real API, but it did not
  converge to a registration beating the no-op baseline on control transforms
  across the hyperparameters tried, and is slow on CPU. We report this rather
  than publish mis-tuned numbers. **PASTE** is excluded too (its internal FGW
  call breaks against POT>=0.9's `line_search`); its partial-OT successor
  **PASTE2** is used.

## Status

**Stop gate: the damage model is implemented and rendered as reference|damaged
pairs with the mask overlaid (`results/damage_examples/*_pairs.png`); the sweep
has not been run.** Environment resolved: all five registration methods
(rigid, PASTE, PASTE2, STalign, GPSA) are available on Python 3.11. Grid set to
~2000 spots, severities {0,2,4,5}, 5 seeds. Review the renders, then green-light
the sweep.

## Authors

Rushil Maniar (Sutura Genomics), Sean Lee (Sutura Genomics), Sunyoung Lee, MD
(MD Anderson Cancer Center). The manuscript source is in `paper/`.
