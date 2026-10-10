import importlib.util
from pathlib import Path
import sqlite3

import pytest

path = Path(__file__).resolve().parents[1] / "scripts/verify_live_outputs.py"
spec = importlib.util.spec_from_file_location("live_verifier", path)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def test_readonly_execution_compares_real_rows_and_rejects_writes(tmp_path):
    database = tmp_path / "farm.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE farm (Total_Horses INTEGER, Working_Horses INTEGER)")
    connection.executemany("INSERT INTO farm VALUES (?, ?)", [(10, 6), (12, 8)])
    connection.commit()
    connection.close()
    original = verifier.execute_readonly(database, "SELECT Total_Horses FROM farm")
    reordered = verifier.execute_readonly(
        database, "SELECT Total_Horses FROM farm ORDER BY Total_Horses DESC"
    )
    alternative = verifier.execute_readonly(database, "SELECT Working_Horses FROM farm")
    assert original["row_count"] == 2
    assert original["bag_sha256"] == reordered["bag_sha256"]
    assert original["ordered_sha256"] != reordered["ordered_sha256"]
    assert original["bag_sha256"] != alternative["bag_sha256"]
    with pytest.raises(ValueError, match="single SELECT"):
        verifier.execute_readonly(database, "DELETE FROM farm")
    with pytest.raises(ValueError, match="single SELECT"):
        verifier.execute_readonly(database, "SELECT * FROM farm; DROP TABLE farm")
    with pytest.raises(ValueError, match="result cap"):
        verifier.execute_readonly(database, "SELECT * FROM farm", row_cap=1)
    assert verifier.execute_readonly(database, "SELECT count(*) FROM farm")["row_count"] == 1


def test_terminal_sql_is_extracted_from_final_agent_turn():
    conversation = [
        {"role": "USER", "content": "Use Total_Horses"},
        {"role": "AGENT", "content": "```sql\nSELECT Total_Horses FROM farm\n```"},
    ]
    assert verifier.terminal_sql(conversation) == "SELECT Total_Horses FROM farm"
    assert (
        verifier.terminal_sql([{"role": "AGENT", "content": "Please clarify the column"}]) is None
    )
