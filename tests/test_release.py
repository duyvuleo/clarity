import gzip
import importlib
import json
from pathlib import Path
import pkgutil

import pytest
import sqlglot

import clarity
from clarity.cli import generate, main
from clarity.datasets import load_annotations, validate_dataset
from clarity.generation.task_flows.components.nodes.parser import SQLParser
from clarity.generation.task_flows.components.nodes.sample_selector import SampleSelector
from clarity.generation.task_flows.components.data_types import ConversationGeneratorTask
from clarity.generation.task_flows.flows.conv_gen_eval_v1 import ConvUtteranceGenEvalsV1
from clarity.models import Llm
from clarity.utils.file_io import load_json, load_jsonl, to_jsonl

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "dataset,annotations,seeds,databases", [("spider", 12098, 7000, 140), ("bird", 7170, 1534, 11)]
)
def test_complete_released_datasets(dataset, annotations, seeds, databases):
    result = validate_dataset(ROOT / "data", dataset)
    assert result["annotations"] == annotations
    assert result["seeds"] == seeds
    assert result["databases"] == databases
    assert sum(result["usecases"].values()) == annotations


def test_all_package_modules_import_without_private_dependencies():
    for module in pkgutil.walk_packages(clarity.__path__, clarity.__name__ + "."):
        if not module.name.endswith("__main__"):
            importlib.import_module(module.name)


@pytest.mark.parametrize(
    "votes,expected",
    [
        ([], False),
        ([True, False], False),
        ([True, True, False], True),
        ([False, False, True], False),
        (["False"], False),
    ],
)
def test_selector_rejects_empty_tied_and_nonboolean_votes(votes, expected):
    assert SampleSelector.majority_vote(votes) is expected


def test_sql_parser_resolves_aliases_and_removes_join_keys():
    sql = sqlglot.parse_one(
        "SELECT s.name, c.concert_name FROM singer s JOIN concert c ON s.singer_id = c.singer_id WHERE s.age >= 30"
    )
    assert SQLParser.extract_columns(sql) == {
        "singer": ["age", "name"],
        "concert": ["concert_name"],
    }
    assert SQLParser.extract_column_value_pairs(sql) == {"singer": [("age", "30")]}


def test_jsonl_and_gzip_io_and_invalid_line(tmp_path):
    path = tmp_path / "records.jsonl"
    to_jsonl(path, [{"question": "café"}])
    assert load_jsonl(path) == [{"question": "café"}]
    path.write_text('{"ok": 1}\n\nnot-json\n')
    with pytest.raises(ValueError, match="records.jsonl:3"):
        load_jsonl(path)
    compressed = tmp_path / "values.json.gz"
    compressed.write_bytes(gzip.compress(b'{"values": [1, 2]}'))
    assert load_json(compressed) == {"values": [1, 2]}


def test_scope_and_task_validation():
    with pytest.raises(ValueError, match="Only spider and bird"):
        load_annotations(ROOT / "data", "unsupported")
    config = json.loads((ROOT / "configs/spider_lexical.json").read_text())
    config["max_examples"] = 0
    with pytest.raises(ValueError):
        ConversationGeneratorTask.model_validate(config)


def test_missing_model_credentials_are_actionable_and_do_not_mutate_config(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = {"vendor": "openai-compatible", "model": "test-model"}
    original = config.copy()
    with pytest.raises(ValueError, match="Set OPENAI_API_KEY"):
        Llm.from_params(config)
    assert config == original


def test_credentials_cannot_be_loaded_from_config():
    with pytest.raises(ValueError, match="environment variables"):
        Llm.from_params({"model": "test-model", "api_key": "fixture"})


@pytest.mark.parametrize(
    "kind",
    [
        "column_confusion",
        "column_ambiguity",
        "value_confusion",
        "single_col_value_ambiguity",
        "multi_cols_value_ambiguity",
    ],
)
def test_real_graph_with_scripted_model(generation_config, scripted_model, kind):
    config = json.loads(generation_config.read_text())
    config["ac_type"] = kind
    if "value_ambiguity" in kind:
        config["similarity_criteria"] = "exact_matches"
    generation_config.write_text(json.dumps(config))
    flow = ConvUtteranceGenEvalsV1(task=str(generation_config), scenario="concise helpful")
    results = list(flow.run())
    assert len(results) == 1
    assert "error" not in results[0], results
    assert results[0]["parser_result"]["selected"]
    assert results[0]["conv_gen_result"]["conversation"]
    assert results[0]["sample_selector"]["outcome"] is True
    assert len(scripted_model) >= 5


@pytest.mark.parametrize("batch", [False, True])
def test_generation_writes_outputs_and_preserves_existing_files(
    generation_config, scripted_model, tmp_path, batch
):
    output = tmp_path / "output"
    receipt = generate(generation_config, output, batch=batch)
    assert receipt["processed"] == 1
    assert receipt["errors"] == 0
    assert receipt["accepted_conversations"] == 1
    assert len(load_jsonl(output / "raw.jsonl")) == 1
    formatted = load_jsonl(output / "concise_helpful.jsonl")
    assert formatted[0]["db_id"] == "concert_singer"
    assert formatted[0]["original_question"] == "What are the names of singers from France?"
    before = (output / "raw.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        generate(generation_config, output)
    assert (output / "raw.jsonl").read_bytes() == before


def test_cli_offline_validation(capsys):
    assert main(["validate", "--data-root", str(ROOT / "data")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert sum(item["annotations"] for item in result) == 19268


@pytest.mark.parametrize("kind,multifacet", [("column_ambiguity", 2), ("column_confusion", 2)])
def test_multifacet_graph(generation_config, scripted_model, kind, multifacet):
    config = json.loads(generation_config.read_text())
    config.update(ac_type=kind, multifacet=multifacet)
    generation_config.write_text(json.dumps(config))
    flow = ConvUtteranceGenEvalsV1(task=str(generation_config), scenario="concise helpful")
    response = next(flow.run())
    assert "error" not in response, response
    assert len(response["parser_result"]["selected"]) == multifacet
    assert response["sample_selector"]["outcome"] is True


def test_semantic_column_generation(generation_config, scripted_model):
    config = json.loads(generation_config.read_text())
    config.update(ac_type="column_ambiguity", similarity_criteria="semantic")
    generation_config.write_text(json.dumps(config))
    flow = ConvUtteranceGenEvalsV1(task=str(generation_config), scenario="concise helpful")
    response = next(flow.run())
    assert "error" not in response, response
    assert response["sample_selector"]["outcome"] is True


def test_malformed_model_output_cannot_be_accepted(generation_config, monkeypatch, tmp_path):
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    model = RunnableLambda(lambda prompt: AIMessage(content="not JSON"))
    monkeypatch.setattr(Llm, "from_params", staticmethod(lambda options: model))
    receipt = generate(generation_config, tmp_path / "invalid-output")
    assert receipt["processed"] == 1
    assert receipt["accepted_conversations"] == 0


def test_all_example_configs_validate():
    paths = sorted((ROOT / "configs").glob("*.json"))
    assert len(paths) == 14
    for path in paths:
        config = ConversationGeneratorTask.model_validate_json(path.read_text())
        assert config.max_examples == 1


def test_value_matching_respects_selected_table_and_missing_columns():
    from types import SimpleNamespace
    from clarity.generation.task_flows.components.nodes.retreiver import SchemaRetriever
    from clarity.generation.task_flows.components.data_types import SettingKey
    from clarity.generation.task_flows.components.signatures import RetrieverResponse

    retriever = SchemaRetriever(output_pydantic_object=RetrieverResponse)
    assert (
        retriever._get_exact_value_matches("France", "missing", {"singer": {"country": ["France"]}})
        == []
    )
    assert retriever._get_range_value_matches(25, "missing", {"singer": {"age": [20, 30]}}) == []
    state = SimpleNamespace(
        example=SimpleNamespace(
            coi=("country", "France"),
            coi_table="singer",
            schema_values={
                "singer": {"country": ["France"]},
                "stadium": {"country": ["France"], "name": ["France"]},
            },
        ),
        parser_result={"selected": [{"singer": "country"}, {"stadium": "country"}]},
    )
    coi, columns = retriever._get_coi_values(
        "France", state, SettingKey.VALUE_AMBIGUITY_MULTI_COLS, SettingKey.EXACT_MATCHES
    )
    assert coi == "country"
    assert columns == ["country"]


def test_partial_existing_outputs_are_preserved(generation_config, scripted_model, tmp_path):
    output = tmp_path / "existing-output"
    output.mkdir()
    (output / "concise_helpful.jsonl").write_text("existing\n")
    with pytest.raises(FileExistsError):
        generate(generation_config, output)
    assert not (output / "raw.jsonl").exists()
    assert (output / "concise_helpful.jsonl").read_text() == "existing\n"
