# Manuscript — Tissue-tear registration benchmark

Overleaf-ready LaTeX source for:

> **Tissue tearing degrades optimal-transport and diffeomorphic registration of
> spatial transcriptomics beyond displacement magnitude: a multi-seed deformation
> benchmark and a supervised graph cross-attention proof-of-concept.**
> Maniar R, Lee S, Lee SS.

## Files

| File | Purpose |
|------|---------|
| `main.tex` | Complete manuscript. Self-contained — bibliography is inline (`thebibliography`), no external `.bib`. |
| `.gitignore` | Ignores LaTeX build artifacts. |
| `fig1_benchmark.png` | Registration error vs tear severity, all method classes. **(add)** |
| `fig2_lodo.png` | Leave-one-donor-out generalization. **(add)** |
| `fig3_magnitude_control.png` | Magnitude-matched control. **(add)** |
| `fig4_architecture.png` | Sutura model architecture. **(add)** |

The four figures are referenced by bare filename, so each PNG must sit in this
same directory (next to `main.tex`). They are **not yet in the repo** — drop them
here before compiling, or LaTeX will error on the missing graphics.

## Build

### Overleaf (recommended)
1. Zip this directory (`main.tex` + the four PNGs), or push it to a Git-linked
   Overleaf project.
2. Set the compiler to **pdfLaTeX** (Menu → Compiler). The document class is
   `extarticle`, which pdfLaTeX handles natively.
3. Compile. No `bibtex`/`biber` pass is needed — references are inline.

### Local
```bash
pdflatex main.tex   # run twice to resolve \ref and \cite cross-references
```
(`latexmk -pdf main.tex` does the two passes automatically.)

## Notes
- 9 pt two-column `extarticle` layout; wide figures use `figure*` (full-width spans).
- `\bibliographystyle{unsrt}` is inert here (manual `thebibliography`); left in place
  for a future switch to an external `.bib`.
- Four bibitems (`memoli2011`, `biancalani2021`, `nature2021`, `pardo2022`) are
  supplementary and appear in the reference list without an in-text `\cite`.
