# Local verification

Validation was performed on October 9, 2026 using Python 3.12.10 on macOS. The dependency versions are pinned in `requirements-lock.txt`. Machine-readable dataset, test and archive evidence is stored under `provenance/`.

**28 tests passed** in the source environment and **28 tests passed** against a fresh wheel installation from outside the source directory. The wheel and source distribution built successfully. Dataset validation covers all 19,268 final annotations and all 8,534 seed records. Original hashes for all 49 selected source artifacts were checked and remained unchanged.

## Scope

The checks exercise every packaged annotation and source seed; all package modules import without private dependencies; the generation graph runs with scripted responses through the real prompt/model/parser chain; sequential and batch output writing preserve existing files. The regression suite includes boolean tie rejection, invalid configurations, malformed model output, multiple facets, semantic ambiguity, compressed I/O and source-table-bound value retrieval.

Network connections are disabled during pytest execution. These are local software tests. Live model generation, model quality, execution accuracy against SQLite databases, original paper result reproduction and institutional publication approval have not been verified by this release work.

The dependency stack retains upstream LangGraph/Pydantic deprecation warnings. They are recorded in the test receipt and do not indicate test failures. Future dependency upgrades should rerun the graph tests before widening the declared version bounds.

The content audit checked the main package and both optional value catalogs. The only reviewed path-pattern exception consists of 199 public BIRD values matched exactly to their source database fields; the catalog hash pins this exception. Other sensitive-pattern rules remain enforced. No claim of exhaustive sensitive-data detection is made.

## Rerun

```bash
python -m pytest --junitxml=provenance/tests.xml
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
clarity validate --data-root data
python scripts/package_release.py
python -m build --no-isolation
```

For the optional catalogs, add `--include-values` to the package audit. To create both data release archives, run `python scripts/package_release.py --output ../releases --include-values` with a new output directory. The archive builder refuses to overwrite existing archives. Its content check covers explicit private-infrastructure/key/token patterns and an allowlist; it is not an exhaustive sensitive-data detector or a legal determination.

The main ZIP includes code and both benchmark subsets. The optional ZIP adds value catalogs. The Python wheel and source distribution contain software and documentation; obtain the benchmark files from the main ZIP. SQLite databases must be acquired separately from the benchmark maintainers.
