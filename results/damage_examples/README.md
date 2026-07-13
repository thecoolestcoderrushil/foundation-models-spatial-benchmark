# Damage examples (visual review before the sweep)

Section: `DLPFC_151507` (4226 spots). Severity 0 = undamaged control; 1..5 increasing. Colours: grey=intact, red=removed, blue=displaced (tear), orange=folded, green=stretched.

- `tear_grid.png`, `tissue_loss_grid.png`, `fold_grid.png`, `stretch_grid.png` - severity (cols) x seed (rows).
- `tear_geometries.png` - straight / curved / branching tear paths.
- `example_artifact.npz` - the emitted ground-truth artifact for one damaged section (mask + per-spot GT displacement + survivor/removed indices).
