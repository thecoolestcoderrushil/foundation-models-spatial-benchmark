# Runtime reliability flags

`runtime_s` is a reported metric. These cells have runtime measurements that should be
treated as unreliable and excluded (or caveated) in any runtime analysis. Correctness
metrics (pitch error, label accuracy) for these cells are **unaffected** — only the timing
is in question. Join `runtime_flags.csv` on `donor,ref,mov,damage_type,severity,seed`.

## `runtime_uncertain` — resume cold-start window (2026-07-14 10:26–10:52Z)

Six cells (`Br5292` tissue_loss sev5 seed2/3/4, fold sev0 seed0, fold sev4 seed0/1), all
methods. The GPSA of the first cell here took **372.3 s** vs a ~78 s median (4.8×), with the
next few cells elevated (101, 107, 114, 126 s) before settling to baseline by ~10:53Z.

Why flagged rather than attributed to warmup:
- **Warmup is falsified as the cause.** The *original* session's first-gpsa-after-cold-start
  (2026-07-13 23:48Z) ran **54.3 s** — a normal cold start. A fresh worker does not
  intrinsically cost 372 s, so cold-start alone cannot explain it.
- The slowdown coincides exactly with heavy diagnostic load from the operator session that
  resumed the sweep (repeated full-CSV `Import-Csv`, git staging/validation dry-runs, failed
  launch attempts, watchdog churn) running concurrently on the same 8-core machine while the
  sweep worker was cold-loading. No other experiment (e.g. hybrid-OT) was running.
- Net: the runtime was measured under uncontrolled, self-induced contention and is not a
  clean measurement. Conservative call: flag, don't trust.

## `runtime_censored` — timeout

`Br5292` tear sev0 seed0, gpsa: hit the `--timeout` budget at **600 s** (`failed=1`,
`reason="timeout >600s"`). Runtime is right-censored at the budget, not a completion time.
Already marked `failed=1` in the CSV; listed here for completeness.
