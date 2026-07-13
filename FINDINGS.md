# Findings — registration degradation under tissue damage

> **Status: pre-sweep.** The damage model is built and rendered
> (`results/damage_examples/`); the degradation sweep has not been run yet
> (stop gate — renders reviewed first). The sections below are the analysis
> template; quantitative results and the ranked leaderboard are filled in after
> `src/run_benchmark.py` completes. Nothing here is fabricated ahead of the data.

## Environment (established)

- **Foundation models** (`results/env_check.md`): all five single-cell FMs
  (scGPT, Geneformer, UCE, scFoundation, Nicheformer) are unusable on this host
  (Windows / CPU / Python 3.14) — scGPT & Geneformer install but fail to import,
  the other three do not install. This is why the benchmark is CPU-only and
  classical-method-only, and it is a reported negative, not a limitation we hide.
- **Registration methods** (`results/methods_env.md`): `rigid`, `PASTE`, `PASTE2`
  available; `STalign` fails to build its pinned numpy on Python 3.14; `GPSA`
  requires a local install (`baselines/GPSA`). Absent methods are marked absent,
  never stubbed.

## Damage model (established)

Four seeded, reproducible damage types with exact per-spot ground truth, each
over severity 0–5. Visual validation: `results/damage_examples/`. The generator
is a standalone package (`damage/`, `pip install -e damage`).

## Degradation leaderboard (TO FILL after sweep)

For each damage type, the severity at which each method's median error crosses
practical thresholds (e.g. 1, 2, 5 spot-pitches), and where each method **breaks**
(failure rate > 50% or degenerate transform).

| method | tear breaks at | tissue_loss breaks at | fold breaks at | stretch breaks at |
|---|---|---|---|---|
| rigid (floor) | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| PASTE | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| PASTE2 | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| GPSA | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| STalign | absent | absent | absent | absent |

## Questions the sweep answers

1. Which damage type is hardest? (Hypothesis: tears/folds — discontinuous,
   topology-changing — break smooth-deformation methods faster than tissue loss,
   which partial-OT PASTE2 is designed for.)
2. Does any method beat the parameter-free rigid floor once damage is severe, or
   do they all collapse toward it?
3. Compute vs robustness: is GPSA's cost justified by damage-robustness, or does
   cheap PASTE2 dominate the Pareto front?
4. Failure modes: crashes vs silent degenerate transforms — which methods fail
   loudly vs quietly?

Negatives will be reported plainly.
