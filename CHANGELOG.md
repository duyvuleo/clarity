# Release-copy changes

## 0.2.1 — October 12, 2026

- Bound release ZIPs, wheels and source distributions to an explicit SHA-256 inventory of reviewed files. Untracked work and caches are excluded; changed reviewed inputs stop publication.
- Replaced private provenance locators with neutral artifact identifiers while retaining source hashes, transformations and attribution.
- Removed workstation ownership and timestamps from source-distribution headers. Runtime algorithms and benchmark records are unchanged.

## 0.2.0 — October 9, 2026

- Wired conventional table-scoped BM25 + dense cosine search with weighted reciprocal-rank fusion into the generation graph. `hybrid` now denotes dense + sparse search; `dense`, `sparse` and existing `llm` modes are available.
- Added configurable Sentence Transformers, OpenAI-compatible, Ollama and OCI embedding adapters. Local model weights are optional dependencies and remain outside release artifacts; authentication stays in environment or existing SDK stores.
- Added optional LLM post-filtering as a separate setting, with schema-bound candidate approval and preserved rank order. Conventional retrieval does not initialize unused retrieval LLMs.
- Added cache fingerprints for embedding/schema changes, dimension and vector validation, corruption detection, and score/rank traces. Added `clarity retrieve` and two public hybrid configuration examples.
- Fixed the alternative-candidate gate to exclude complete, case-normalized column names; value ambiguity excludes the relevant source value or column. Removed obsolete private-retriever helper paths.
- Added ranking, alpha-endpoint, scope-isolation, cache-rebuild, provider-contract and real graph integration tests. Local pretrained embedding verification is recorded separately from mocked remote-provider contracts.

## 0.1.1 — October 9, 2026

- Added an optional authenticated Codex CLI model adapter, using ephemeral, read-only sessions, structured responses, tool-use rejection and credential-free call receipts.
- Fixed the live retrieval context handoff: the selected table schema and target column are now set before the retriever calls its model. Added prompt-context assertions to prevent regression.
- Added a read-only local SQL verifier for generated conversations, schema grounding and helpful-dialogue result comparison. Alternative dialogue targets are reported separately from the inherited parent SQL.
- Retained initial live rejections and the amended protocol; live smoke-test results are recorded with the release provenance rather than interpreted as a benchmark score.

## 0.1.0 — October 9, 2026

- Extracted the CLARITY generation pipeline and shared flow utilities into an installable `clarity` package. Replaced working-directory-dependent imports with package imports.
- Removed the private retrieval service and the unused eager database-value cache. Spider/BIRD retrieval uses the provided benchmark catalogs.
- Replaced infrastructure-specific model adapters and credentials files with an environment-configured OpenAI-compatible adapter. Model configuration is copied before use, preserving caller state.
- Added bounded configuration validation, portable data paths, UTF-8 JSONL, gzip value loading, and actionable input errors. Column runs avoid loading the large value catalogs.
- Added a CLI for validation, statistics and generation. Output files use exclusive creation and managed handles; generation records a receipt and returns a failure status for runtime errors.
- Fixed sample-selection ties returning a truthy string instead of a boolean. Empty and non-boolean votes reject samples. Fixed error-response construction to create a response instance instead of modifying its class.
- Fixed graph node names that collided with the `sample_selector` state field. Retained the source graph's routing, including its bypass of conversation-evaluation nodes.
- Bound value matching to the selected source table, handled missing column catalogs, and matched formatted outputs to their exact source seed IDs and database IDs. Batch execution now respects the configured concurrency limit.
- Removed duplicate model-field and prompt-key declarations, updated Pydantic descriptions and allowed the historical column/value intermediate forms. Replaced string-enum list membership with explicit value-setting tuples.
- Removed the global random seed set during module import; the CLI supplies a seed for new runs. Normalized Python formatting and removed an unnecessary terminal dependency.
- Retained final annotations and seeds byte-for-byte. Filtered mixed schema, column and value catalogs to the selected Spider/BIRD database IDs. Kept value catalogs as optional release assets.
- Added offline dataset checks, parser/voting regressions, configuration checks, and tests of real LangGraph execution with scripted model responses, including multiple facets, semantic ambiguity, sequential/batch writing and malformed-output rejection.

The source folders are preserved. These changes do not reconstruct the original evaluated prediction files, evaluation masks, private evaluators, or author database snapshot. The local BIRD count discrepancy is documented rather than silently modified.
