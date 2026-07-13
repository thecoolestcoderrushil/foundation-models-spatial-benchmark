# PHASE 0 - Foundation-model environment check

_Host: Windows, CPU-only, tested on Python 3.12 via uv isolated venvs._

| model | installable | importable | status | time (s) |
|---|---|---|---|---|
| `scgpt` | True | False | **installed_but_broken** | 165.1 |
| `geneformer` | True | False | **installed_but_broken** | 83.6 |
| `uce` | False | False | **not_installable** | 3.7 |
| `scfoundation` | False | False | **not_installable** | 4.5 |
| `nicheformer` | False | False | **not_installable** | 0.6 |

## Exact outcome per model

### `scgpt` - installed_but_broken
- _PyPI package; needs a pretrained checkpoint + gene-vocab download, flash-attn (CUDA) for the fast path, and IPython/anndata pins._
- **Detail:** import failed: C:\Users\karti\spatial-foundation-benchmark\_fmtest\scgpt\Lib\site-packages\scgpt\model\model.py:21: UserWarning: flash_attn is not installed
  warnings.warn("flash_attn is not installed")
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "C:\Users\karti\spatial-foundation-benchmark\_fmtest\scgpt\Lib\site-packages\scgpt\__init__.py", line 18, in <module>
    from . import model, tokenizer, scbank, utils, tasks
  File "C:\Users\karti\spatial-foundation-benchmark\_fmtest\scgpt\Lib\site-packages\scgpt\model\__init__.py", line 8, in <module>
    from .generation_model import *
  File "C:\Users\karti\spatial-foundation-benchmark\_fmtest\scgpt\Lib\site-packages\scgpt\model\generation_model.py", line 21, in <module>
    from ..utils import map_raw_id_to_vocab_id
  File "C:\Users\karti\spatial-foundation-benchmark\_fmtest\scgpt\Lib\site-packages\scgpt\utils\__init__.py", line 1, in <module>
    from .util import *
  File "C:\Users\karti\spatial-foundation-benchmark\_fmtest\scgpt\Lib\site-packages\scgpt\utils\util.py", line 16, in <module>
    from IPython import get_ipython
ModuleNotFoundError: No module named 'IPython'

### `geneformer` - installed_but_broken
- _Not on PyPI; distributed as a HuggingFace git-LFS repo with tokenizer + median-expression dictionaries + model weights._
- **Detail:** import failed: Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "C:\Users\karti\spatial-foundation-benchmark\_fmtest\geneformer\Lib\site-packages\geneformer\__init__.py", line 17, in <module>
    from . import (
  File "C:\Users\karti\spatial-foundation-benchmark\_fmtest\geneformer\Lib\site-packages\geneformer\collator_for_classification.py", line 12, in <module>
    from transformers import (
ImportError: cannot import name 'SpecialTokensMixin' from 'transformers' (C:\Users\karti\spatial-foundation-benchmark\_fmtest\geneformer\Lib\site-packages\transformers\__init__.py)

### `uce` - not_installable
- _Not a pip package; a GitHub research repo run via eval scripts, needs a multi-GB ESM2 protein-embedding file + model checkpoint._
- **Detail:** install exit 2: Using Python 3.12.13 environment at: _fmtest\uce
   Updating https://github.com/snap-stanford/UCE.git (HEAD)
    Updated https://github.com/snap-stanford/UCE.git (9c416007be15ad6753dc84af4468c1dc10421ab9)
error: C:\Users\karti\AppData\Local\uv\cache\git-v0\checkouts\29c06e6bd1284d36\9c41600 does not appear to be a Python project, as neither `pyproject.toml` nor `setup.py` are present in the directory

### `scfoundation` - not_installable
- _Not on PyPI; GitHub repo, weights gated behind a request form, designed for CUDA GPUs._
- **Detail:** install exit 2: Using Python 3.12.13 environment at: _fmtest\scfoundation
error: C:\Users\karti\AppData\Local\uv\cache\git-v0\checkouts\e7a04259cdde48cf\397631c does not appear to be a Python project, as neither `pyproject.toml` nor `setup.py` are present in the directory

### `nicheformer` - not_installable
- _theislab package; needs weights from HuggingFace/Zenodo and is built for GPU spatial-omics pretraining._
- **Detail:** install exit 1: Using Python 3.12.13 environment at: _fmtest\nicheformer
  Ã— No solution found when resolving dependencies:
  â•°â”€â–¶ Because nicheformer was not found in the package registry and you
      require nicheformer, we can conclude that your requirements are
      unsatisfiable.

## Summary

**Foundation models usable in this environment: NONE.**

Baselines below are pure-Python / scvi-tools and always run, so the benchmark has featurizers to compare either way:
- `pca` (TruncatedSVD), `hvg` (highly-variable-gene selection), `scvi` (scVI latent), `scvi_batch` (scVI + batch correction).