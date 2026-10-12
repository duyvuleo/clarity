# Local verification

Release 0.2.1 adds publication controls. `PUBLICATION.json` pins reviewed repository files;
release ZIPs and PEP 517 builds verify these bytes and stage only listed files. Source
distributions use neutral ownership metadata. Editable installs continue to use the
developer's checkout and are for local development.

After reviewing and staging intentional release changes, refresh the inventory explicitly
with `python publication.py --from-index`, then stage `PUBLICATION.json`. Builds never
refresh the inventory automatically. The software sdist excludes benchmark data and can
rebuild a wheel without Git. Optional value catalogs retain their separate pinned hashes.
`source-manifest.json` retains the prepared-source hashes recorded for 0.2.0;
`PUBLICATION.json` pins the current release bytes.

Release 0.2.0 has 49 passing regression tests in both the source and installed-wheel environments and adds conventional hybrid schema retrieval. Its [dedicated report](provenance/hybrid-retrieval/REPORT.md) records live local embeddings, two live graph cases, and database checks. Remote embedding providers have mocked contract coverage only. The earlier 0.1.1 evidence below is historical.

Validation was performed on October 9, 2026 using Python 3.12.10 on macOS. The dependency versions are pinned in `requirements-lock.txt`. Machine-readable dataset, test and archive evidence is stored under `provenance/`.

Release 0.1.1 has **33 passing regression tests** in both the source environment and a separately installed wheel, including Codex transport checks and a read-only SQLite verifier. Dataset validation covers all 19,268 final annotations and all 8,534 seed records. Original hashes for all 49 selected source artifacts were checked and remained unchanged. The release receipts record source, installed-wheel and archive validation separately. Release 0.1.0 previously passed 28 tests; its evidence is retained in the earlier archives.

## Scope

The checks exercise every packaged annotation and source seed; all package modules import without private dependencies; the generation graph runs with scripted responses through the real prompt/model/parser chain; sequential and batch output writing preserve existing files. The regression suite includes boolean tie rejection, invalid configurations, malformed model output, multiple facets, semantic ambiguity, compressed I/O and source-table-bound value retrieval.

Network connections are disabled during pytest execution. In a separate authenticated live run, 30 Codex CLI calls generated five conversations from one Spider seed and five from one BIRD seed. Local schema checks passed, eight terminal SELECTs executed in read-only SQLite, and all four fully helpful conversations matched their parent-query result bags. Two other conversations explicitly declined SQL without sufficient clarification. Five diagnostic candidate queries also executed, with three alternatives differing from their parent results.

See [the live report](provenance/live-validation/REPORT.md) and its call receipts, raw outputs and local-verification ledger. Initial rejected cases are retained: live inference revealed missing schema/column context in retrieval, which was repaired and regression-tested before rerunning the same frozen inputs.

This is a two-case smoke test of single-facet semantic column ambiguity, using the same model for generation and evaluation and one judge per check. Other modes remain unverified with live models. Exact backend model identity is unavailable from the CLI JSON event stream. Semantic adjudication, whole-benchmark execution accuracy, original paper result reproduction and institutional publication approval have not been established.

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
