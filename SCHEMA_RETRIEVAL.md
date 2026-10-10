# Conventional schema retrieval

Release 0.2.0 wires table-scoped dense cosine search and BM25 keyword search into the generation graph. Retrieval uses local NumPy indexes and does not require a running vector database. `hybrid` now means dense + BM25; earlier releases used that label for an incomplete candidate/LLM-filter path. Existing `llm` configurations retain LLM retrieval.

## Configure and run

```bash
python -m pip install -e '.[hybrid]'
clarity retrieve --config configs/spider_hybrid.json \
  --db-id farm --table farm --query Total_Horses
```

`retrieve` loads benchmark catalogs and performs retrieval without initializing generation LLMs. The first local embedding request downloads the selected public model into `.cache/embedding-models/`; the example pins its revision. Set `local_files_only: true` after caching the model for offline runs. Model weights and vector caches are excluded from release archives and Git.

Generation uses the same backend:

```bash
clarity generate --config configs/spider_hybrid.json --output outputs/spider-hybrid
```

Configure the generation/evaluation model roles as usual before running `generate`. The hybrid configuration has no retrieval LLM by default.

Configuration excerpt:

```json
{
  "retrieval_type": "hybrid",
  "llm_retrieval": [],
  "schema_retrieval": {
    "embedding": {
      "provider": "sentence-transformers",
      "model": "sentence-transformers/all-MiniLM-L6-v2",
      "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
      "device": "cpu",
      "batch_size": 32,
      "cache_folder": "../.cache/embedding-models"
    },
    "sparse": {"k1": 1.5, "b": 0.75},
    "fusion": "rrf",
    "alpha": 0.5,
    "top_k": 25,
    "candidate_k": 50,
    "rrf_k": 60,
    "cache_dir": "../.cache/schema-index",
    "llm_filter": false
  }
}
```

`data_root`, model caches and index caches resolve relative to the configuration file. `top_k` limits returned columns; `candidate_k` limits each ranking before fusion and must be at least `top_k`. `min_dense_similarity` optionally removes dense candidates below a cosine threshold. Dense retrieval alone uses `retrieval_type: dense`; BM25 alone uses `sparse`. In hybrid mode, `alpha: 0` selects only sparse results and requires no embedding model, while `alpha: 1` selects only dense results.

The weighted reciprocal-rank score is `alpha / (rrf_k + dense_rank) + (1 - alpha) / (rrf_k + sparse_rank)`. A missing ranking contributes zero. Only positive BM25 matches enter the sparse ranking. Ranking ties are broken by column name. This implements a transparent local fusion rule; it does not claim identical scores to a particular Weaviate version.

## Embedding providers

| Provider | Configuration | Runtime requirement |
| --- | --- | --- |
| `sentence-transformers` | `model`, `revision`, `device`, optional prefixes and dimensions | Install `.[hybrid]`; local encoder |
| `openai-compatible` | `model`, optional `dimensions`, `api_key_env`, `base_url_env` | Credentials and endpoint in environment |
| `ollama` | `model`, `base_url_env`, optional dimensions | Existing Ollama service; default localhost port 11434 |
| `oci` | `model`, `oci_profile`, optional dimensions, compartment/endpoint environment names | Install `.[oci-embeddings]`; existing local OCI authentication |

Use `document_prefix` and `query_prefix` when required by an embedding model. Index and query embeddings use the same configured provider and model. OCI explicitly uses `SEARCH_DOCUMENT` for indexed schema text and `SEARCH_QUERY` for queries. Credentials cannot be supplied as embedding configuration fields; they are read only at request time from environment variables or existing SDK credential stores. They are not included in index metadata.

The local pretrained encoder has been exercised on Spider/BIRD schemas. Remote provider request contracts are unit-tested with mocked transports; remote embedding services are not claimed as live-verified. Relevant primary references: [Sentence Transformers](https://sbert.net/docs/package_reference/sentence_transformer/model.html), [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [OpenAI embeddings](https://developers.openai.com/api/docs/guides/embeddings), [Ollama embeddings](https://docs.ollama.com/api/embed), and [OCI EmbedTextDetails](https://docs.oracle.com/en-us/iaas/tools/python/latest/api/generative_ai_inference/models/oci.generative_ai_inference.models.EmbedTextDetails.html).

## Scope, filtering and cache validity

One document represents one physical schema column, including its identifier, table and declared type. Identifier tokenization splits underscores and camel case. Every search is restricted to its requested database and table before embedding or ranking. Database values are not indexed by this column-retrieval backend.

Set `llm_filter: true` and provide `llm_retrieval` model configurations to apply the existing lexical/semantic classification prompt after ranking. The filter can approve only retrieved physical columns and preserves their ranking order. Unknown or hallucinated names do not enter the candidate group. The downstream ambiguity gate requires an alternative besides the complete, case-normalized target column.

The dense cache fingerprint includes the embedding provider/model/revision, prefixes, requested dimensions, endpoint identity hash, source schema and ordered column documents. Changing these settings produces a new index instead of reusing an incompatible one. Old generated cache files are preserved. Loading checks metadata, row counts, finite values and nonzero vectors; query dimensions must match the index. A corrupted cache causes an actionable error. For an API service that replaces weights under the same model name, change `revision` to identify the new space before reuse.

Raw generation examples retain `retrieval_hits` with dense/sparse ranks, individual scores and fused scores. These explain retrieval decisions; they are not semantic validity labels. See `provenance/hybrid-retrieval/` for local embedding and graph verification evidence.
