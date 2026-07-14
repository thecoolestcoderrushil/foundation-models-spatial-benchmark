# Findings - registration degradation under tissue damage

_Computed from `results/benchmark_results.csv` (31 method-cells). Numbers are mean per-cell median spot-pitch error over serial pairs and seeds, excluding failed cells._

## Median registration error (spot-pitches) by severity


### tear

| method | sev 0 | sev 2 | sev 4 | breaks at |
|---|---|---|---|---|
| `gpsa` | 3.11 | 3.31 | 5.69 | 4 |
| `paste2` | 4.02 | 3.84 | 3.92 | never |
| `rigid` | 2.92 | 2.94 | 4.97 | never |

## Is the rigid floor competitive?

- **tear** @ sev 2: rigid 2.94 vs paste2 3.84 pitch -> rigid <= paste2

## Failures (crashes + degenerate/timeout)

| method | damage | severity | failure rate |
|---|---|---|---|
| - | - | - | none observed |

## Interpretation (from the numbers above)

- The parameter-free rigid ICP floor is at least as good as PASTE2 at low damage (sev 2) on: tear. Expensive OT does not buy accuracy there.
- On **tear**, `gpsa` breaks earliest (median error > 5 pitch by severity 4).

## Excluded / caveats

- **STalign**: real LDDMM API wired (py3.11 + numpy<2) but did not converge beating no-op on control transforms across tried hyperparameters; excluded rather than shown mis-tuned.
- **PASTE**: POT>=0.9 `line_search` API break in its FGW; PASTE2 used.
- **Damage realism**: primary tears are interior cracks (closed outline); the boundary-reaching `tear_edge` condition probes the gap.
- Moving section = damaged self + known rigid misalignment (isolates damage; omits inter-section biology).