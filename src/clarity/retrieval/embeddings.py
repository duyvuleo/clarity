"""Lazy embedding adapters. Authentication stays in environment or local SDK stores."""

import hashlib
import json
import os
from pathlib import Path
from typing import Protocol

import numpy as np

from clarity.retrieval.config import EmbeddingConfig


class EmbeddingProvider(Protocol):
    def identity(self) -> dict: ...
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class ConfiguredEmbeddings:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None
        self._client = None

    def _local_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ValueError(
                    "Install clarity-nl2sql[hybrid] for sentence-transformers embeddings"
                ) from None
            self._model = SentenceTransformer(
                self.config.model,
                revision=self.config.revision,
                device=self.config.device,
                cache_folder=self.config.cache_folder,
                trust_remote_code=False,
                local_files_only=self.config.local_files_only,
            )
        return self._model

    def identity(self):
        config = self.config
        revision = config.revision
        if config.provider == "sentence-transformers":
            model = self._local_model()
            revision = getattr(getattr(model[0], "auto_model", None), "config", None)
            revision = getattr(revision, "_commit_hash", None) or config.revision
            if revision is None:
                from huggingface_hub import try_to_load_from_cache

                cached = try_to_load_from_cache(
                    config.model, "config.json", cache_dir=config.cache_folder
                )
                if isinstance(cached, str):
                    revision = Path(cached).parent.name
        endpoint = (
            os.environ.get(
                config.base_url_env if config.provider != "oci" else config.service_endpoint_env, ""
            )
            if config.provider != "sentence-transformers"
            else ""
        )
        # Credentials and resource identifiers are never included in persisted metadata.
        model_name = (
            config.model
            if not config.model.startswith("ocid" + "1.")
            else hashlib.sha256(config.model.encode()).hexdigest()
        )
        return {
            "provider": config.provider,
            "model": model_name,
            "revision": revision,
            "dimensions": config.dimensions,
            "document_prefix": config.document_prefix,
            "query_prefix": config.query_prefix,
            "endpoint_sha256": hashlib.sha256(endpoint.encode()).hexdigest() if endpoint else None,
        }

    def _oci_client(self):
        if self._client is None:
            try:
                import oci
            except ImportError:
                raise ValueError(
                    "Install clarity-nl2sql[oci-embeddings] for OCI embeddings"
                ) from None
            options = oci.config.from_file(self.config.oci_config_file, self.config.oci_profile)
            signer = None
            if options.get("security_token_file"):
                token = Path(options["security_token_file"]).expanduser().read_text().strip()
                key = oci.signer.load_private_key_from_file(options["key_file"])
                signer = oci.auth.signers.SecurityTokenSigner(token, key)
            endpoint = os.environ.get(self.config.service_endpoint_env)
            self._client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                options,
                signer=signer,
                service_endpoint=endpoint,
                timeout=(10, self.config.timeout),
            )
        return self._client

    def _remote_batch(self, texts, is_query):
        config = self.config
        try:
            if config.provider == "openai-compatible":
                from openai import OpenAI

                key = os.environ.get(config.api_key_env)
                if not key:
                    raise ValueError(f"Set {config.api_key_env} before requesting embeddings")
                if self._client is None:
                    self._client = OpenAI(
                        api_key=key,
                        base_url=os.environ.get(config.base_url_env),
                        timeout=config.timeout,
                        max_retries=1,
                    )
                arguments = {"model": config.model, "input": texts, "encoding_format": "float"}
                if config.dimensions is not None:
                    arguments["dimensions"] = config.dimensions
                response = self._client.embeddings.create(**arguments)
                ordered = sorted(response.data, key=lambda item: item.index)
                if [item.index for item in ordered] != list(range(len(texts))):
                    raise ValueError("Embedding response indices differ from the input batch")
                return [item.embedding for item in ordered]
            if config.provider == "ollama":
                import httpx

                endpoint = os.environ.get(config.base_url_env, "http://localhost:11434").rstrip("/")
                arguments = {"model": config.model, "input": texts, "truncate": False}
                if config.dimensions is not None:
                    arguments["dimensions"] = config.dimensions
                response = httpx.post(
                    endpoint + "/api/embed", json=arguments, timeout=config.timeout
                )
                response.raise_for_status()
                return response.json()["embeddings"]
            if config.provider == "oci":
                import oci

                compartment = os.environ.get(config.compartment_id_env)
                if not compartment:
                    raise ValueError(
                        f"Set {config.compartment_id_env} before requesting embeddings"
                    )
                arguments = {
                    "inputs": texts,
                    "compartment_id": compartment,
                    "serving_mode": oci.generative_ai_inference.models.OnDemandServingMode(
                        model_id=config.model
                    ),
                    "input_type": "SEARCH_QUERY" if is_query else "SEARCH_DOCUMENT",
                    "truncate": "NONE",
                    "is_echo": False,
                }
                if config.dimensions is not None:
                    arguments["output_dimensions"] = config.dimensions
                details = oci.generative_ai_inference.models.EmbedTextDetails(**arguments)
                return self._oci_client().embed_text(details).data.embeddings
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(
                f"{config.provider} embedding request failed ({type(error).__name__}); check endpoint, model and local authentication"
            ) from None
        raise ValueError("Unsupported embedding provider")

    def _embed(self, texts, is_query):
        config = self.config
        prefix = config.query_prefix if is_query else config.document_prefix
        prepared = [prefix + text for text in texts]
        if config.provider == "sentence-transformers":
            values = self._local_model().encode(
                prepared,
                batch_size=config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            if config.dimensions is not None and values.shape[1] != config.dimensions:
                raise ValueError("Configured dimensions do not match the local embedding model")
            result = np.asarray(values, dtype=np.float32)
            if result.ndim != 2 or len(result) != len(texts):
                raise ValueError("Embedding response does not match the input batch")
            return result
        rows = []
        for start in range(0, len(prepared), config.batch_size):
            rows.extend(self._remote_batch(prepared[start : start + config.batch_size], is_query))
        result = np.asarray(rows, dtype=np.float32)
        if result.ndim != 2 or len(result) != len(texts) or not np.isfinite(result).all():
            raise ValueError("Embedding response does not match the input batch")
        if config.dimensions is not None and result.shape[1] != config.dimensions:
            raise ValueError("Embedding provider returned different dimensions than configured")
        return result

    def embed_documents(self, texts):
        return self._embed(texts, False)

    def embed_query(self, text):
        return self._embed([text], True)[0]


def create_embedding_provider(config):
    return ConfiguredEmbeddings(config)
