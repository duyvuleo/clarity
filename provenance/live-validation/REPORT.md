# Live CLARITY generation and local verification

On October 9, 2026, the CLARITY generation graph produced five conversations from one frozen Spider seed and five from one frozen BIRD seed using live, authenticated Codex CLI model responses. All 30 model calls in the corrected attempt completed successfully, with no tool-use events.

| Local source case | Generated conversations | Terminal SELECTs executed | Fully helpful result checks |
| --- | ---: | ---: | --- |
| Spider, farm, seed 20 | 5 | 4 | 2/2 matched source results |
| BIRD, card_games, seed 357 | 5 | 4 | 2/2 matched source results |

All generated schema references exist in their local databases. Source questions and inherited parent SQL were preserved; original seed-file indices are recorded separately. Both database hashes match the pinned local manifest. Eight emitted terminal SELECT queries executed successfully in read-only SQLite. The two remaining conversations explicitly declined to generate SQL because the user did not clarify the column.

The parent Spider query returned eight rows; the BIRD count query returned one scalar result row. Five annotation-derived candidate queries were also executed. Two Spider alternatives and one BIRD alternative returned different result bags from their parent queries, demonstrating a local execution consequence of the selected ambiguity. These derived alternatives are diagnostic constructions, not newly adjudicated gold references.

## Live-discovered defect and repair

The first live attempt made two retrieval calls, but both cases were rejected before generation: the retriever had not been given the selected table schema or target column. Scripted outputs had concealed that context gap. Release 0.1.1 sets both fields before retrieval, adds assertions over the actual retrieval prompt, and retains the original rejected attempts alongside the successful rerun. No source data or original research code was modified.

## Verification boundary

This is a two-case software smoke test of single-facet semantic column ambiguity, selected from existing local annotations. It uses the same model for generation and evaluation and a single judge per check. It is not an independent semantic adjudication, a corpus acceptance rate, an execution-accuracy benchmark score, or reproduction of the paper's ensemble, masks or predictions. Other generation modes remain covered by local regression tests rather than live tests.

The installed Codex CLI rejected the configured gpt-6.1-sol model name. Its working default was used with low reasoning effort. The exact backend model identity is not exposed in the captured JSON event stream and remains unavailable; the CLI version and requested settings are recorded. Sessions were ephemeral and read-only; shell and web search were disabled, and any tool-use event would have failed the model adapter. Credentials were neither copied into this package nor recorded in receipts.

Parent SQL is used as a reference only for the concise and verbose helpful dialogues that select the original column. Partially helpful and non-helpful dialogues are reported separately: an alternative candidate or an explicit refusal can be appropriate. Matching the parent execution result does not prove semantic equivalence across database instances.

See SUMMARY.json, local-verification.json, cases.json, model-calls.jsonl and the outputs folder for inspectable evidence. PROTOCOL.md records the frozen cases and the amendment after the initial rejection.
