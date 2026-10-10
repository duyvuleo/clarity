# CLARITY Spider/BIRD data card

This release contains 19,268 final conversational annotation records: 12,098 Spider and 7,170 BIRD. It also contains the local source seed files: 7,000 Spider records and 1,534 BIRD records. File hashes and exact counts are recorded in the manifests.

## Record fields

| Field | Meaning |
| --- | --- |
| `db_id` | Source benchmark database identifier |
| `question` | Generated ambiguous or unanswerable request |
| `original_question` | Parent natural-language question |
| `expectedQuery` | Inherited parent SQL; not necessarily the terminal dialogue target |
| `usecase` | Historical ambiguity/confusion, facet and conversation-type label |
| `ac_term` | Generated pivot term(s) |
| `selected_column` | Selected column(s), or column/value structures for value cases |
| `similar_columns` | Candidate interpretation metadata when present |
| `conversation` | Ordered `USER`/`AGENT` turns with text in `content` |

Additional historical fields such as `dataset`, `template`, `completion`, `schema`, `initialState`, `currentState`, `history` and `synonymsStatus` are preserved, including null values. Schema catalogs are provided separately. Record order and contents are preserved.

## Use and limitations

The intended uses are research on ambiguity detection, schema-level localization, clarification, unanswerability and conversational NL2SQL. Generated annotations and model-derived conversations may contain errors. A software validity check does not establish semantic correctness.

The local seed file names do not establish an official training/development/test partition. In particular, the 7,000-record Spider seed file spans 140 database IDs. Keep experiments grouped by source question and database to avoid conversation variants leaking across splits. The release does not invent a new official split.

The local BIRD file has 7,170 records; the paper reports 7,171. This discrepancy is retained and reported rather than patched with an invented instance. The original evaluated masks, predictions and reference/evaluation scripts are not recovered by packaging this snapshot. The original author database snapshot identity remains unverified.

`expectedQuery` is inherited from the parent request. A dialogue may ask to choose any candidate or choose a different candidate. Do not treat it as an independently validated terminal SQL target for every conversation. Raw source SQL is not proof of every permissible interpretation, and execution on one database instance is not proof of semantic equivalence.

The value catalogs contain source-benchmark values, potentially including personal information already present in those public datasets. Preserve the upstream terms and use normal research data handling practices. Only approved benchmark IDs are retained from the mixed source catalogs.

The BIRD catalog includes school website links and public code-community text containing local-looking paths. All 199 unique values flagged by the path check were matched exactly against the corresponding fields in the downloaded official BIRD development databases. They are retained as benchmark content. The pinned content review is recorded in `provenance/benchmark-content-review.json`; it does not exempt infrastructure identifiers or credential patterns from the release check.
