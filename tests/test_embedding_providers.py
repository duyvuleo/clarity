from types import SimpleNamespace
import sys
import uuid

import numpy as np
import pytest

from clarity.retrieval.config import EmbeddingConfig
from clarity.retrieval.embeddings import ConfiguredEmbeddings


def test_openai_batch_order_dimensions_and_no_credentials_in_identity(monkeypatch):
    key = uuid.uuid4().hex
    monkeypatch.setenv("EMBED_TEST_KEY", key)
    observed = []

    def create(**arguments):
        observed.append(arguments)
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[index + 1, 2])
                for index in reversed(range(len(arguments["input"])))
            ]
        )

    backend = ConfiguredEmbeddings(
        EmbeddingConfig(
            provider="openai-compatible",
            model="configured-model",
            dimensions=2,
            api_key_env="EMBED_TEST_KEY",
            document_prefix="passage: ",
            query_prefix="query: ",
        )
    )
    backend._client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    assert backend.embed_documents(["power", "toughness"]).tolist() == [[1, 2], [2, 2]]
    assert observed[0]["dimensions"] == 2 and observed[0]["input"] == [
        "passage: power",
        "passage: toughness",
    ]
    assert backend.embed_query("power").tolist() == [1, 2]
    assert observed[1]["input"] == ["query: power"]
    assert key not in str(backend.identity())


def test_openai_rejects_missing_or_duplicate_response_indices(monkeypatch):
    monkeypatch.setenv("EMBED_TEST_KEY", uuid.uuid4().hex)
    backend = ConfiguredEmbeddings(
        EmbeddingConfig(provider="openai-compatible", model="model", api_key_env="EMBED_TEST_KEY")
    )
    backend._client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(
                data=[
                    SimpleNamespace(index=0, embedding=[1, 2]),
                    SimpleNamespace(index=0, embedding=[1, 2]),
                ]
            )
        )
    )
    with pytest.raises(ValueError, match="indices"):
        backend.embed_documents(["one", "two"])


def test_ollama_transport_uses_configured_model_and_does_not_truncate(monkeypatch):
    import httpx

    requests = []

    def post(url, **kwargs):
        requests.append((url, kwargs["json"]))
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"embeddings": [[1, 2] for _ in kwargs["json"]["input"]]},
        )

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setenv("EMBED_TEST_URL", "http://localhost:11434")
    backend = ConfiguredEmbeddings(
        EmbeddingConfig(
            provider="ollama", model="configured-model", base_url_env="EMBED_TEST_URL", dimensions=2
        )
    )
    assert backend.embed_query("power").shape == (2,)
    assert requests[0] == (
        "http://localhost:11434/api/embed",
        {"model": "configured-model", "input": ["power"], "truncate": False, "dimensions": 2},
    )


def test_oci_uses_document_and_query_input_types(monkeypatch):
    captured = []

    class Details:
        def __init__(self, **kwargs):
            self.arguments = kwargs

    fake_oci = SimpleNamespace(
        generative_ai_inference=SimpleNamespace(
            models=SimpleNamespace(
                OnDemandServingMode=lambda **kwargs: kwargs, EmbedTextDetails=Details
            )
        )
    )
    monkeypatch.setitem(sys.modules, "oci", fake_oci)
    monkeypatch.setenv("OCI_EMBED_TEST_COMPARTMENT", uuid.uuid4().hex)
    backend = ConfiguredEmbeddings(
        EmbeddingConfig(
            provider="oci",
            model="configured-model",
            compartment_id_env="OCI_EMBED_TEST_COMPARTMENT",
            dimensions=512,
        )
    )

    def embed(details):
        captured.append(details.arguments)
        return SimpleNamespace(
            data=SimpleNamespace(
                embeddings=np.ones((len(details.arguments["inputs"]), 512)).tolist()
            )
        )

    backend._client = SimpleNamespace(embed_text=embed)
    backend.embed_documents(["power"])
    backend.embed_query("power")
    assert [row["input_type"] for row in captured] == ["SEARCH_DOCUMENT", "SEARCH_QUERY"]
    assert all(row["output_dimensions"] == 512 and row["truncate"] == "NONE" for row in captured)
