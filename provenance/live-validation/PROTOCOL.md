# Live Codex smoke-test protocol

Run the unchanged generation graph on one frozen Spider seed and one frozen BIRD seed, selected from the local packaged snapshot for an existing single-column semantic ambiguity setting. Use one Codex CLI default model for every generator/evaluator role, low reasoning effort, sequential execution and a single judge per check. Retain all generation errors and rejections; this is a two-case software smoke test, not a benchmark rate or the paper ensemble.

Check source database hashes, selected/candidate schema bindings, source-question/parent-SQL preservation, conversation structure, and read-only execution of parent and terminal SQL. For concise and verbose helpful dialogues, compare full result bags with parent SQL. Other dialogue styles may choose an alternative candidate, so record their execution separately without treating parent SQL as universal terminal gold. Do not claim semantic adjudication or historical result reproduction.

Model calls use ephemeral Codex sessions in a read-only sandbox, with shell and web search disabled; reject any tool-use event. Never read or copy account credentials. The CLI's working default model is used because the installed CLI rejected the configured gpt-6.1-sol name. Exact backend model identity is not reported by the JSON event stream and remains unavailable.

## Amendment after initial live retrieval

The first two live calls received missing table-schema and selected-column context, and both cases were rejected before generation. Those outputs are retained in the initial outputs folder. Fixed the missing context handoff without changing graph routing and added prompt-content assertions to the regression fixture. The same frozen seeds are rerun in attempt2; retain both attempts.
