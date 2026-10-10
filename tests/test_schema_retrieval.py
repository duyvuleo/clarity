import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from clarity.retrieval.config import EmbeddingConfig, RetrievalConfig
from clarity.retrieval.index import BM25, SchemaIndex, tokenize
from clarity.generation.task_flows.components.edges.retriever_decision import RetrieverDecision
from clarity.generation.task_flows.components.signatures import RetrieverResponse


class FixedEmbeddings:
    def __init__(self, name="test-A", dimension=2):
        self.name, self.dimension, self.document_calls = name, dimension, 0

    def identity(self):
        return {"provider": "test", "model": self.name}

    def embed_documents(self, texts):
        self.document_calls += 1
        vectors = []
        for text in texts:
            vector = (
                [0, 1]
                if "identifier toughness" in text
                else [1, 0]
                if "identifier power" in text
                else [1, 1]
            )
            vectors.append(vector if self.dimension == 2 else vector + [1])
        return np.array(vectors)

    def embed_query(self, text):
        return np.array([0, 1] if self.dimension == 2 else [0, 1, 1])


def index(config=None, embedding=None, mode="hybrid", changed_schema=False):
    ddl = "CREATE TABLE cards (power TEXT, toughness TEXT, cardCode TEXT)"
    if changed_schema:
        ddl = ddl.replace("power TEXT", "power INTEGER")
    schemas = {
        "card_games": {"cards": ddl},
        "other_db": {"cards": "CREATE TABLE cards (privateField TEXT)"},
    }
    columns = {
        "card_games": {"cards": ["power", "toughness", "cardCode"]},
        "other_db": {"cards": ["privateField"]},
    }
    return SchemaIndex(
        schemas, columns, config or RetrievalConfig(top_k=3, candidate_k=3), embedding, mode
    )


def test_identifier_tokenization_and_bm25_real_keyword_ranking():
    assert tokenize("Total_Horses cardCode") == ["total", "horses", "card", "code"]
    scores = BM25(["rare identifier", "common word", "common common word"]).scores("rare")
    assert scores[0] > 0 and scores[1] == scores[2] == 0
    repeated = BM25(["power power", "power other"]).scores("power")
    assert repeated[0] > repeated[1]
    assert BM25(["a", "b"]).scores("unseen").tolist() == [0, 0]


def test_fusion_uses_both_rankings_and_alpha_endpoints():
    hybrid = index(embedding=FixedEmbeddings()).search("power", "card_games", "cards")
    assert hybrid[0]["column"] == "power"
    assert hybrid[0]["dense_rank"] == 3 and hybrid[0]["sparse_rank"] == 1
    assert hybrid[0]["score"] == pytest.approx(0.5 / 63 + 0.5 / 61)
    assert hybrid[1]["column"] == "toughness"
    dense = index(RetrievalConfig(alpha=1, top_k=3, candidate_k=3), FixedEmbeddings()).search(
        "power", "card_games", "cards"
    )
    assert dense[0]["column"] == "toughness" and all(row["sparse_rank"] is None for row in dense)
    sparse = index(RetrievalConfig(alpha=0, top_k=3, candidate_k=3)).search(
        "power", "card_games", "cards"
    )
    assert [row["column"] for row in sparse] == ["power"]
    assert sparse[0]["dense_rank"] is None


def test_database_and_table_scope_precede_embedding_calls():
    embedding = FixedEmbeddings()
    backend = index(embedding=embedding)
    with pytest.raises(ValueError, match="database"):
        backend.search("power", "missing", "cards")
    with pytest.raises(ValueError, match="table"):
        backend.search("power", "card_games", "missing")
    assert embedding.document_calls == 0
    results = backend.search("power", "card_games", "CARDS")
    assert all(row["db_id"] == "card_games" and row["table"] == "cards" for row in results)
    assert not any(row["column"] == "privateField" for row in results)


def test_cache_reuse_and_rebuild_on_model_and_schema_changes(tmp_path):
    config = RetrievalConfig(cache_dir=str(tmp_path), top_k=3, candidate_k=3)
    first = FixedEmbeddings()
    backend = index(config, first)
    backend.search("power", "card_games", "cards")
    first_fingerprint = backend.last_index_metadata["fingerprint"]
    assert first.document_calls == 1 and len(list(tmp_path.glob("*.npz"))) == 1
    second = FixedEmbeddings()
    cached = index(config, second)
    cached.search("power", "card_games", "cards")
    assert second.document_calls == 0 and cached.last_index_metadata["cache_loaded"] is True
    changed = index(config, FixedEmbeddings("test-B", 3))
    changed.search("power", "card_games", "cards")
    assert changed.last_index_metadata["fingerprint"] != first_fingerprint
    assert changed.last_index_metadata["dimensions"] == 3
    schema_change = index(config, FixedEmbeddings(), changed_schema=True)
    schema_change.search("power", "card_games", "cards")
    assert schema_change.last_index_metadata["fingerprint"] != first_fingerprint
    assert len(list(tmp_path.glob("*.npz"))) == 3


def test_invalid_vectors_and_corrupt_cache_fail_without_silent_fallback(tmp_path):
    class Bad(FixedEmbeddings):
        def embed_documents(self, texts):
            return np.zeros((len(texts), 2))

    with pytest.raises(ValueError, match="zero-length"):
        index(embedding=Bad()).search("power", "card_games", "cards")

    class WrongDimension(FixedEmbeddings):
        def embed_query(self, text):
            return np.ones(3)

    with pytest.raises(ValueError, match="dimensions"):
        index(embedding=WrongDimension()).search("power", "card_games", "cards")
    config = RetrievalConfig(cache_dir=str(tmp_path), top_k=3, candidate_k=3)
    index(config, FixedEmbeddings()).search("power", "card_games", "cards")
    path = next(tmp_path.glob("*.npz"))
    path.write_bytes(b"invalid index")
    with pytest.raises(ValueError, match="cache is invalid"):
        index(config, FixedEmbeddings()).search("power", "card_games", "cards")
    assert path.read_bytes() == b"invalid index"


def test_index_metadata_tracks_scope_when_reusing_cached_vectors():
    backend = index(embedding=FixedEmbeddings())
    backend.search("power", "card_games", "cards")
    first = backend.last_index_metadata["fingerprint"]
    backend.search("privateField", "other_db", "cards")
    assert backend.last_index_metadata["fingerprint"] != first
    backend.search("power", "card_games", "cards")
    assert backend.last_index_metadata["fingerprint"] == first
    assert backend.last_index_metadata["documents"] == 3


@pytest.mark.parametrize(
    "group,expected", [(["POWER"], False), (["power", "toughness"], True), ([], False)]
)
def test_candidate_gate_excludes_complete_case_normalized_target(group, expected):
    state = SimpleNamespace(
        retriever_result={0: RetrieverResponse(group=group, coi="power")},
        example=SimpleNamespace(cois=["power"], retriever_decision=None),
    )
    config = {"configurable": {"llm": SimpleNamespace(ac_type="column_ambiguity")}}
    assert RetrieverDecision()(state, config) is expected


def test_configuration_validates_embeddings_and_fusion_parameters():
    from clarity.generation.task_flows.components.data_types import ConversationGeneratorTask

    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/spider_hybrid.json").read_text())
    config["schema_retrieval"]["embedding"] = None
    with pytest.raises(ValueError, match="embedding"):
        ConversationGeneratorTask.model_validate(config)
    with pytest.raises(ValueError):
        RetrievalConfig(alpha=1.1)
    with pytest.raises(ValueError, match="candidate_k"):
        RetrievalConfig(top_k=10, candidate_k=5)
    with pytest.raises(ValueError):
        EmbeddingConfig(provider="openai-compatible", model="model", api_key="fixture")


@pytest.mark.parametrize("filtered", [False, True])
def test_conventional_retrieval_is_used_in_actual_generation_graph(
    generation_config, scripted_model, monkeypatch, filtered
):
    from clarity.retrieval import factory
    from clarity.generation.task_flows.flows.conv_gen_eval_v1 import ConvUtteranceGenEvalsV1

    class ForFixture(FixedEmbeddings):
        def embed_documents(self, texts):
            self.document_calls += 1
            return np.array([[1, 0] if "identifier age" in text else [0, 1] for text in texts])

    monkeypatch.setattr(factory, "create_embedding_provider", lambda options: ForFixture())
    config = json.loads(generation_config.read_text())
    config.update(
        ac_type="column_ambiguity",
        retrieval_type="hybrid",
        schema_retrieval={
            "embedding": {"provider": "sentence-transformers", "model": "fixture"},
            "top_k": 3,
            "candidate_k": 3,
            "llm_filter": filtered,
        },
    )
    generation_config.write_text(json.dumps(config))
    if filtered:
        from clarity.generation.task_flows.components.nodes.retreiver import SchemaRetriever
        from clarity.generation.task_flows.components.signatures import (
            RetrieverClassificationResponse,
        )

        def classify(self, state, task, model):
            assert set(state.example.subset_columns) == {"name", "country", "age"}
            return RetrieverClassificationResponse(
                response=[
                    {"column_name": "name", "outcome": "valid"},
                    {"column_name": "country", "outcome": "valid"},
                    {"column_name": "hallucinated", "outcome": "valid"},
                ]
            )

        monkeypatch.setattr(SchemaRetriever, "LLM_call", classify)
    flow = ConvUtteranceGenEvalsV1(task=str(generation_config), scenario="concise helpful")
    response = next(flow.run())
    assert "error" not in response, response
    assert response["sample_selector"]["outcome"] is True
    assert response["example"]["retrieval_hits"]["0"]
    assert not any(keys == {"group", "coi"} for keys in scripted_model)
    if filtered:
        assert "hallucinated" not in response["retriever_result"]["0"]["group"]
