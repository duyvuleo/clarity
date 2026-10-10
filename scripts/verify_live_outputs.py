"""Check generated conversations against packaged schemas and read-only SQLite.

The parent query is a reference only for helpful dialogues that request the original
column. Other dialogue styles are reported separately rather than forced to that gold.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time

import sqlglot
from sqlglot import exp


def execute_readonly(database, sql, row_cap=5000, timeout=5):
    statements = sqlglot.parse(sql, read="sqlite")
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise ValueError("Only a single SELECT statement is allowed")
    connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        start = time.monotonic()
        connection.set_progress_handler(lambda: int(time.monotonic() - start > timeout), 10000)
        cursor = connection.execute(sql)
        rows = cursor.fetchmany(row_cap + 1)
        if len(rows) > row_cap:
            raise ValueError("Query exceeds the declared result cap")
        encoded = [json.dumps(row, ensure_ascii=False, default=str) for row in rows]
        bag = sorted(Counter(encoded).items())
        return {
            "row_count": len(rows),
            "column_count": len(cursor.description),
            "ordered_sha256": hashlib.sha256(json.dumps(encoded).encode()).hexdigest(),
            "bag_sha256": hashlib.sha256(json.dumps(bag).encode()).hexdigest(),
        }
    finally:
        connection.close()


def terminal_sql(conversation):
    for turn in reversed(conversation):
        if turn.get("role") != "AGENT":
            continue
        text = turn.get("content", "").strip()
        fenced = re.findall(r"```(?:sql|sqlite)?\s*(.*?)```", text, re.I | re.S)
        candidates = fenced + [text]
        for candidate in candidates:
            if re.match(r"^SELECT\b", candidate.strip(), re.I):
                try:
                    sqlglot.parse_one(candidate, read="sqlite")
                    return candidate.strip()
                except sqlglot.errors.ParseError:
                    pass
        return None
    return None


def verify_case(raw_file, output_directory, database, schemas, source_seed):
    responses = [json.loads(line) for line in Path(raw_file).open() if line.strip()]
    if len(responses) != 1 or responses[0].get("error"):
        raise ValueError("Expected one successful live graph response")
    response = responses[0]
    example = response["example"]
    if (example["db_id"], example["question"], example["expected_query"]) != (
        source_seed["db_id"],
        source_seed["question"],
        source_seed["expectedQuery"],
    ):
        raise ValueError("Live result differs from its frozen source seed")
    tables = {
        name.lower(): [column[1].lower() for column in columns] for name, columns in schemas.items()
    }
    selected = response["parser_result"]["selected"]
    bindings = []
    for item in selected:
        table, column = next(iter(item.items()))
        if table.lower() not in tables or str(column).lower() not in tables[table.lower()]:
            raise ValueError("Selected source column does not exist in the local database")
        bindings.append((table, column))
    for index, retrieval in response["term_gen_result"].items():
        table = bindings[int(index)][0].lower()
        for column in retrieval.get("group", []):
            if column.lower() not in tables[table]:
                raise ValueError("Generated candidate column does not exist in the selected table")
    parent = execute_readonly(database, source_seed["expectedQuery"])
    parent_tree = sqlglot.parse_one(source_seed["expectedQuery"], read="sqlite")
    if (
        len(list(parent_tree.find_all(exp.Table))) != 1
        or parent_tree.find(exp.Join)
        or parent_tree.find(exp.Subquery)
    ):
        raise ValueError(
            "Candidate derivation is limited to single-table parent queries without joins or subqueries"
        )
    candidate_executions = []
    for index, retrieval in response["term_gen_result"].items():
        table, original_column = bindings[int(index)]
        for candidate in dict.fromkeys([original_column] + retrieval.get("group", [])):
            variant = sqlglot.parse_one(source_seed["expectedQuery"], read="sqlite")
            replacements = 0
            for column in variant.find_all(exp.Column):
                if column.name.lower() == original_column.lower():
                    column.set("this", exp.to_identifier(candidate))
                    replacements += 1
            if not replacements:
                raise ValueError("Selected source column was not found in parent SQL")
            query = variant.sql(dialect="sqlite")
            result = execute_readonly(database, query)
            candidate_executions.append(
                {
                    "table": table,
                    "column": candidate,
                    "derived_sql": query,
                    "execution": result,
                    "same_result_as_parent": result["bag_sha256"] == parent["bag_sha256"],
                }
            )
    checks = []
    for path in sorted(Path(output_directory).glob("*.jsonl")):
        if path.name == "raw.jsonl":
            continue
        rows = [json.loads(line) for line in path.open() if line.strip()]
        for row in rows:
            turns = row.get("conversation", [])
            valid_turns = bool(turns) and all(
                isinstance(turn, dict)
                and turn.get("role") in {"USER", "AGENT"}
                and isinstance(turn.get("content"), str)
                for turn in turns
            )
            if not valid_turns:
                raise ValueError("Malformed generated conversation")
            if turns[-1]["role"] != "AGENT" or any(
                turn["role"] != ("USER" if index % 2 == 0 else "AGENT")
                for index, turn in enumerate(turns)
            ):
                raise ValueError("Conversation roles do not alternate from USER to AGENT")
            if turns[0]["role"] != "USER" or turns[0]["content"] != row["question"]:
                raise ValueError("Conversation changed the generated initial request")
            if row["expectedQuery"] != source_seed["expectedQuery"]:
                raise ValueError("Formatted record changed parent SQL")
            query = terminal_sql(turns)
            check = {
                "scenario": path.stem,
                "turn_count": len(turns),
                "structure_valid": True,
                "terminal_sql_present": query is not None,
            }
            if query:
                result = execute_readonly(database, query)
                check.update(
                    terminal_sql=query,
                    terminal_execution=result,
                    matches_parent_result=result["bag_sha256"] == parent["bag_sha256"],
                )
            if path.stem in {"concise_helpful", "verbose_helpful"}:
                check["parent_reference_applies"] = True
                check["helpful_sql_check_passed"] = (
                    query is not None and check.get("matches_parent_result") is True
                )
            else:
                check["parent_reference_applies"] = False
            checks.append(check)
    return {
        "db_id": source_seed["db_id"],
        "selected_bindings": bindings,
        "schema_grounding_passed": True,
        "source_seed_preserved": True,
        "parent_execution": parent,
        "annotation_derived_candidate_executions": candidate_executions,
        "sample_selector_accepted": response["sample_selector"]["outcome"] is True,
        "conversation_checks": checks,
        "helpful_checks_passed": all(
            check["helpful_sql_check_passed"]
            for check in checks
            if check["parent_reference_applies"]
        ),
        "helpful_checks": sum(check["parent_reference_applies"] for check in checks),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--database-manifest", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads((args.run / "cases.json").read_text())
    databases = json.loads(args.database_manifest.read_text())["databases"]
    results = []
    for case in cases:
        db = next(
            row
            for row in databases
            if row["dataset"].lower() == case["dataset"] and row["db_id"] == case["db_id"]
        )
        with Path(db["path"]).open("rb") as handle:
            database_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        if database_hash != db["sha256"]:
            raise ValueError("Local database hash differs from the pinned manifest")
        connection = sqlite3.connect(Path(db["path"]).resolve().as_uri() + "?mode=ro", uri=True)
        try:
            names = [
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            ]
            schema = {
                name: list(
                    connection.execute('PRAGMA table_info("' + name.replace('"', '""') + '")')
                )
                for name in names
            }
        finally:
            connection.close()
        output = args.run / "outputs" / case["dataset"]
        result = verify_case(output / "raw.jsonl", output, db["path"], schema, case["seed"])
        result.update(
            dataset=case["dataset"],
            source_seed_index=case["source_seed_index"],
            source_database_sha256=db["sha256"],
            source_database_hash_verified=True,
            author_database_snapshot_identity_verified=False,
        )
        results.append(result)
    (args.run / "local-verification.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
