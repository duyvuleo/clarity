import gzip
import json
from pathlib import Path
import re

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
import pytest

from clarity.models import Llm

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def generation_config(tmp_path):
    """A small fixture based on Spider's public concert_singer schema."""
    data = tmp_path / "data/spider"
    data.mkdir(parents=True)
    seed = {
        "db_id": "concert_singer",
        "dataset": None,
        "question": "What are the names of singers from France?",
        "expectedQuery": "SELECT name FROM singer WHERE country = 'France'",
    }
    schema = {
        "concert_singer": {"singer": "CREATE TABLE singer (name TEXT, country TEXT, age INTEGER)"}
    }
    columns = {"concert_singer": {"singer": ["name", "country", "age"]}}
    # A deliberately shared test value exercises the multiple-column value branch.
    values = {
        "concert_singer": {
            "singer": {"name": ["France", "Bob"], "country": ["France", "England"], "age": [25, 30]}
        }
    }
    for filename, payload in [("schemas.json", schema), ("columns.json", columns)]:
        (data / filename).write_text(json.dumps(payload))
    (data / "seeds.jsonl").write_text(json.dumps(seed) + "\n")
    (data / "values.json.gz").write_bytes(gzip.compress(json.dumps(values).encode()))
    config = json.loads((ROOT / "configs/spider_column_confusion.json").read_text())
    config["data_root"] = "data"
    path = tmp_path / "task.json"
    path.write_text(json.dumps(config))
    return path


@pytest.fixture
def scripted_model(monkeypatch):
    """Return schema-shaped text through the real prompt/model/parser chain."""
    calls = []

    def respond(prompt):
        text = prompt.to_string()
        matches = re.findall(r"Here is the output schema:\s*```\s*(.*?)\s*```", text, re.S)
        assert matches, "Every model request must include the structured-output schema"
        keys = set(json.loads(matches[-1])["properties"])
        calls.append(keys)
        if "conversation" in keys:
            response = {
                "conversation": [
                    {"role": "USER", "content": "What are their performer details?"},
                    {"role": "AGENT", "content": "Which column do you mean?"},
                    {"role": "USER", "content": "Use name."},
                ]
            }
        elif "ac_query" in keys:
            response = {
                "nl_query": "What are their names?",
                "ac_query": "What are their performer details?",
            }
        elif "category" in keys:
            response = {"category": "untranslatable", "reason": "Scripted test assessment"}
        elif "terms" in keys:
            response = {"terms": ["performer"], "reason": "Scripted test assessment"}
        elif "outcome" in keys:
            response = {
                "outcome": "yes" if "yes' or 'no" in text else "valid",
                "reason": "Scripted test assessment",
            }
        elif "term" in keys:
            response = {"term": "performer", "reason": "Scripted test generation"}
            if "group" in keys:
                response["group"] = ["name", "country"]
        elif "group" in keys:
            response = {"group": ["name", "country"], "coi": "name", "voi": "France"}
        else:
            raise AssertionError(f"Uncovered model schema: {keys}")
        return AIMessage(content=json.dumps(response))

    model = RunnableLambda(respond)
    monkeypatch.setattr(Llm, "from_params", staticmethod(lambda options: model))
    return calls
