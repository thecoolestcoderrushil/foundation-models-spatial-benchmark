"""
PHASE 0 - Environment check for single-cell foundation models.

For each foundation model we attempt a real install into an ISOLATED throwaway
venv (so a broken install cannot corrupt the baseline environment that actually
runs the benchmark) and then attempt to import it. Every outcome is recorded
verbatim - "installable+importable", "installed but broken", or "not
installable" - with the exact error text. Nothing is silently skipped.

We deliberately test on Python 3.12 (best compatibility for these packages) via
`uv`, which hardlinks wheels from a shared cache so repeated torch installs are
cheap. Results are written incrementally to results/env_check.md and .json.

Usage:
  python src/env_check.py            # run all model probes
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
WORK = ROOT / "_fmtest"
UV = shutil.which("uv") or "uv"
PY = "3.12"
# Time-boxed: a foundation model that needs a multi-GB git-LFS / weight download
# to even install is, for practical purposes, "not usable in this environment".
# We record that honestly rather than hang the run.
INSTALL_TIMEOUT = 420   # seconds per model install

# Each model: how the community actually ships it. Foundation models are rarely
# clean PyPI packages - most are GitHub repos + separate multi-GB weight files.
MODELS = [
    dict(name="scgpt",
         pip=["scgpt"],
         imp="scgpt",
         note="PyPI package; needs a pretrained checkpoint + gene-vocab download, "
              "flash-attn (CUDA) for the fast path, and IPython/anndata pins."),
    dict(name="geneformer",
         pip=["geneformer @ git+https://huggingface.co/ctheodoris/Geneformer"],
         imp="geneformer",
         note="Not on PyPI; distributed as a HuggingFace git-LFS repo with "
              "tokenizer + median-expression dictionaries + model weights."),
    dict(name="uce",
         pip=["git+https://github.com/snap-stanford/UCE.git"],
         imp="uce",
         note="Not a pip package; a GitHub research repo run via eval scripts, "
              "needs a multi-GB ESM2 protein-embedding file + model checkpoint."),
    dict(name="scfoundation",
         pip=["git+https://github.com/biomap-research/scFoundation.git"],
         imp="scfoundation",
         note="Not on PyPI; GitHub repo, weights gated behind a request form, "
              "designed for CUDA GPUs."),
    dict(name="nicheformer",
         pip=["nicheformer"],
         imp="nicheformer",
         note="theislab package; needs weights from HuggingFace/Zenodo and is "
              "built for GPU spatial-omics pretraining."),
]


def run(cmd, timeout):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr, time.time() - t0
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or ""), f"TIMEOUT after {timeout}s", time.time() - t0
    except Exception as e:  # pragma: no cover
        return 1, "", f"{type(e).__name__}: {e}", time.time() - t0


def last_err(stdout, stderr, n=1600):
    """The most informative tail of an install/import failure."""
    txt = (stderr or "").strip() or (stdout or "").strip()
    return txt[-n:] if txt else "(no output)"


def probe(m):
    name = m["name"]
    venv = WORK / name
    if venv.exists():
        shutil.rmtree(venv, ignore_errors=True)
    rec = dict(model=name, note=m["note"], installable=False, importable=False,
               status="not_installable", detail="", install_s=0.0)

    rc, so, se, dt = run([UV, "venv", "--python", PY, str(venv)], 120)
    if rc != 0:
        rec["detail"] = f"venv creation failed: {last_err(so, se)}"
        return rec
    py = venv / "Scripts" / "python.exe"
    if not py.exists():
        py = venv / "bin" / "python"

    # Install (with a CPU torch already available via the shared index so the FM
    # doesn't drag in a CUDA build). Time-boxed.
    rc, so, se, dt = run(
        [UV, "pip", "install", "--python", str(py),
         "--index-strategy", "unsafe-best-match", *m["pip"]],
        INSTALL_TIMEOUT)
    rec["install_s"] = round(dt, 1)
    if rc != 0:
        rec["status"] = "not_installable"
        rec["detail"] = f"install exit {rc}: {last_err(so, se)}"
        return rec
    rec["installable"] = True

    # Import test in the isolated interpreter.
    rc, so, se, dt = run([str(py), "-c", f"import {m['imp']}; print('OK')"], 300)
    if rc == 0 and "OK" in so:
        rec["importable"] = True
        rec["status"] = "installable_importable"
        rec["detail"] = "installed and importable"
    else:
        rec["status"] = "installed_but_broken"
        rec["detail"] = f"import failed: {last_err(so, se)}"
    return rec


def write_reports(records):
    RESULTS.mkdir(parents=True, exist_ok=True)
    order = {m["name"]: i for i, m in enumerate(MODELS)}
    records = sorted(records, key=lambda r: order.get(r["model"], 99))
    (RESULTS / "env_check.json").write_text(json.dumps(records, indent=2), "utf-8")
    lines = ["# PHASE 0 - Foundation-model environment check\n",
             f"_Host: Windows, CPU-only, tested on Python {PY} via uv isolated venvs._\n",
             "| model | installable | importable | status | time (s) |",
             "|---|---|---|---|---|"]
    for r in records:
        lines.append(f"| `{r['model']}` | {r['installable']} | {r['importable']} | "
                     f"**{r['status']}** | {r['install_s']} |")
    lines.append("\n## Exact outcome per model\n")
    for r in records:
        lines.append(f"### `{r['model']}` - {r['status']}")
        lines.append(f"- _{r['note']}_")
        lines.append(f"- **Detail:** {r['detail']}\n")
    working = [r["model"] for r in records if r["importable"]]
    lines.append("## Summary\n")
    lines.append(f"**Foundation models usable in this environment: "
                 f"{', '.join(working) if working else 'NONE'}.**")
    lines.append("\nBaselines below are pure-Python / scvi-tools and always run, "
                 "so the benchmark has featurizers to compare either way:")
    lines.append("- `pca` (TruncatedSVD), `hvg` (highly-variable-gene selection), "
                 "`scvi` (scVI latent), `scvi_batch` (scVI + batch correction).")
    (RESULTS / "env_check.md").write_text("\n".join(lines), "utf-8")


def _load_done():
    """Resume support: models already recorded (survives reaping between probes)."""
    p = RESULTS / "env_check.json"
    if p.exists():
        try:
            return {r["model"]: r for r in json.loads(p.read_text("utf-8"))}
        except Exception:
            return {}
    return {}


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    done = _load_done()
    records = list(done.values())
    for m in MODELS:
        if m["name"] in done:
            print(f"=== {m['name']} already probed ({done[m['name']]['status']}); skip ===",
                  flush=True)
            continue
        print(f"=== probing {m['name']} ===", flush=True)
        try:
            rec = probe(m)
        except Exception as e:
            rec = dict(model=m["name"], note=m["note"], installable=False,
                       importable=False, status="probe_error",
                       detail=f"{type(e).__name__}: {e}", install_s=0.0)
        records.append(rec)
        print(f"  -> {rec['status']}: {rec['detail'][:200]}", flush=True)
        write_reports(records)                 # incremental
    # clean the throwaway venvs (keep disk sane); reports already written
    shutil.rmtree(WORK, ignore_errors=True)
    print("PHASE 0 done. see results/env_check.md", flush=True)


if __name__ == "__main__":
    main()
