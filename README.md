# CLARITY

CLARITY generates and studies conversational ambiguity and unanswerability in natural language to SQL. This release packages the local Spider and BIRD annotation snapshots and the generation pipeline described in [the CLARITY paper](https://arxiv.org/abs/2604.22313).

The released files contain **12,098 Spider conversations** and **7,170 BIRD conversations**, with 7,000 and 1,534 source seed records respectively. Schema and column catalogs cover 140 Spider databases and 11 BIRD databases. These are counts of the packaged files; they are not reconstructed paper evaluation splits.

## Install and check

Use Python 3.11 or later. Python 3.12 is the locally verified environment. From this release directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e '.[dev]'
clarity validate --data-root data
python -m pytest
```

`clarity stats --data-root data --dataset spider` reports use-case counts. Dataset loading, validation and the tests run without credentials or model access. The lock file records the dependency versions used for local validation; the declared dependency bounds are in `pyproject.toml`.

## Data

| File in `data/spider/` or `data/bird/` | Contents |
| --- | --- |
| `annotations.jsonl` | Byte-identical final CLARITY conversation annotations |
| `seeds.jsonl` | Byte-identical local SQL/question seeds used by the generation code |
| `schemas.json`, `columns.json` | Catalogs filtered to the selected benchmark database IDs |
| `values.json.gz` | Optional compressed database-value catalog for value experiments |

The main archive contains code, annotations, seeds, schemas and columns. Extract the separate `clarity-values-0.2.0.zip` beside it to add the value catalogs. Column generation and annotation analysis do not require that asset. The BIRD value catalog is large; allow several gigabytes of memory when loading it. The value catalogs are ignored by Git so they can be distributed as release assets rather than oversized repository files.

SQLite databases are not bundled. Obtain them from the [Spider](https://yale-lily.github.io/spider) and [BIRD](https://bird-bench.github.io/) maintainers if you need SQL execution. The value catalogs support generation; they do not substitute for executable databases.

See [DATA_CARD.md](DATA_CARD.md) for fields and interpretation limits, and [provenance/source-manifest.json](provenance/source-manifest.json) for exact source hashes and transformations.

## Generate new conversations

Conventional dense + BM25 schema retrieval is available through [the hybrid configurations and guide](SCHEMA_RETRIEVAL.md). Embedding provider/model, fusion weight, top-k and optional LLM filtering are configurable. Use `clarity retrieve` to inspect rankings without generation-model credentials.

Each config in `configs/` starts with one seed to keep trial runs small. Set `max_examples` to a larger integer, or `null` for the full local seed file. Set `multifacet` to 1, 2 or 3. Choose lexical/semantic column ambiguity, column confusion, value confusion, or single/multiple-column value ambiguity. In this historical implementation, “confusion” denotes an unanswerability setting.

Set your own model name and credential in your shell, and optionally `OPENAI_BASE_URL` for an OpenAI-compatible endpoint. Keep credentials out of configuration files. Then run:

```bash
clarity generate --config configs/spider_lexical.json --output outputs/spider-trial
```

Generation invokes the configured endpoint and may incur its normal charges. Each model role is independently configurable; the examples use one user-selected model for all roles. New runs are new experiments and do not reproduce the paper's original model ensemble or evaluated predictions. `--seed 10` controls local sequential SQL target sampling; model outputs and parallel scheduling remain nondeterministic.

The output directory contains raw intermediate results, formatted scenario files, and a receipt with processed, accepted and error counts. Existing output files are preserved. Generation errors cause a nonzero exit status. The two historical “not helpful” variants are retained as separate files. Conversation-evaluation nodes are retained, but the source graph bypasses them and uses the sample selector after conversation generation; this release preserves that routing.

### Use an authenticated Codex CLI

Alternatively, install [Codex CLI](https://learn.chatgpt.com/docs/non-interactive-mode) and sign in locally. Set each model role to `{"vendor": "codex", "reasoning_effort": "low"}` in a copy of the relevant configuration. Use one entry per evaluator list for a small trial. An optional `model` selects a CLI-supported model name, and `receipt_path` records prompt/response hashes, timings and usage without credentials or raw CLI logs.

The adapter calls `codex exec` in an ephemeral, read-only session with shell and web search disabled. It rejects any tool-use event and parses the model answer through the same structured-output chain as the other provider. Saved CLI authentication stays in Codex's existing credential store. This is intended for local research runs; it does not add account credentials or Codex authentication to the public repository or CI.

## Verification and reuse

Tests cover every packaged annotation, package imports, parser behavior, rejection voting, invalid inputs, compressed I/O, model configuration, and actual LangGraph execution with scripted model responses. They verify local software behavior rather than semantic quality of a live model or the paper's reported results. See [VERIFICATION.md](VERIFICATION.md) for the measured validation boundary.

Release 0.1.1 also includes a [live Codex smoke test](provenance/live-validation/REPORT.md): one frozen Spider seed and one frozen BIRD seed produced ten conversations. Eight emitted SELECT queries executed against pinned local SQLite instances, and all four fully helpful dialogues matched the parent-query result bags. Two insufficient-clarification dialogues explicitly declined to produce SQL. This live run exposed and fixed a missing schema/column context handoff. It covers two single-facet semantic-column cases, with one model and one judge; it does not establish whole-benchmark performance or semantic correctness of every annotation.

Code and original software documentation use **MIT**, as selected for this release. The Spider/BIRD-derived records and catalogs use **CC BY-SA 4.0** with upstream attribution. The MIT software license does not replace the data terms. See [DATA_LICENSE.md](DATA_LICENSE.md) and [NOTICE.md](NOTICE.md).

Please cite CLARITY and the relevant source benchmark when using the release. [CITATION.cff](CITATION.cff) contains the CLARITY reference. [CHANGELOG.md](CHANGELOG.md) documents release-copy fixes and refactoring.

Release 0.2.0 includes [hybrid retrieval verification](provenance/hybrid-retrieval/REPORT.md), using pinned pretrained embeddings in the live Spider/BIRD generation graph. The preceding LLM-retrieval validation remains available for comparison.
