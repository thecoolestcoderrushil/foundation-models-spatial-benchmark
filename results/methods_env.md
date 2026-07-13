# Registration methods - environment availability

_Host: Windows, CPU-only, Python 3.14 main env._

| method | available | detail |
|---|---|---|
| `rigid` | True |  |
| `paste` | True | paste-bio importable |
| `paste2` | True | paste2 importable |
| `stalign` | False | ModuleNotFoundError: No module named 'STalign' |
| `gpsa` | False | ModuleNotFoundError: No module named 'gpsa' |

- `rigid` is implemented in-repo (no dependency), the parameter-free floor.
- Absent methods are recorded here and skipped by the harness; their APIs are never stubbed.