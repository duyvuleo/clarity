# Hybrid schema retrieval verification

Release 0.2.0 executes BM25 and dense cosine search locally, combines their rankings using weighted reciprocal-rank fusion, and passes the resulting physical columns into the existing generation graph. LLM filtering is optional and was disabled for this run.

The pretrained CPU encoder was sentence-transformers/all-MiniLM-L6-v2, revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41, with 384-dimensional vectors. All retrieved column names were checked against the local SQLite schemas. Working_Horses ranked third for the Spider Total_Horses query; toughness ranked sixth for the BIRD power query. Source query columns had contributions from both dense and sparse rankings.

The same two frozen source seeds from the preceding live test were run through the actual graph with conventional hybrid retrieval. The graph generated five conversations per seed. All 28 Codex generator/evaluator calls completed without tool use. Eight emitted terminal SELECT queries executed on the pinned local databases, and all four fully helpful conversations matched their source-query result bags. The remaining two dialogues explicitly declined SQL without sufficient clarification.

The regression suite checks fusion endpoints and both ranking contributions, database/table isolation, no-LLM retrieval, optional schema-bound filtering, model/schema cache changes, dimension mismatch, corrupted caches, complete target-column exclusion, and mocked remote provider request contracts. Local model inference is live-verified; OpenAI-compatible, Ollama and OCI embedding transports have only mocked contract coverage.

This is a two-case single-facet semantic-column smoke test with one generation/evaluation model and one judge. It does not establish semantic correctness of every interpretation, whole-corpus recall, benchmark execution accuracy, or reproduction of published results. Exact Codex backend model identity remains unavailable from the JSON event stream. Existing data files and original research source folders are preserved.

Inspect embedding-smoke.json, summary.json, model-calls.jsonl, local-verification.json and outputs/ for the measured evidence. Configuration and migration instructions are in SCHEMA_RETRIEVAL.md.
