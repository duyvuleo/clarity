from typing import Literal

from pydantic import BaseModel, Field, model_validator


class EmbeddingConfig(BaseModel, extra="forbid"):
    provider: Literal["sentence-transformers", "openai-compatible", "ollama", "oci"]
    model: str = Field(min_length=1)
    revision: str | None = None
    dimensions: int | None = Field(default=None, ge=1)
    batch_size: int = Field(default=32, ge=1, le=128)
    device: str = "cpu"
    cache_folder: str | None = None
    local_files_only: bool = False
    document_prefix: str = ""
    query_prefix: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    timeout: float = Field(default=120, gt=0)
    oci_profile: str = "DEFAULT"
    oci_config_file: str = "~/.oci/config"
    compartment_id_env: str = "OCI_COMPARTMENT_ID"
    service_endpoint_env: str = "OCI_GENAI_ENDPOINT"


class SparseConfig(BaseModel, extra="forbid"):
    k1: float = Field(default=1.5, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)


class RetrievalConfig(BaseModel, extra="forbid"):
    embedding: EmbeddingConfig | None = None
    sparse: SparseConfig = Field(default_factory=SparseConfig)
    fusion: Literal["rrf"] = "rrf"
    alpha: float = Field(default=0.5, ge=0, le=1)
    top_k: int = Field(default=25, ge=1)
    candidate_k: int = Field(default=50, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    min_dense_similarity: float | None = Field(default=None, ge=-1, le=1)
    cache_dir: str | None = None
    llm_filter: bool = False

    @model_validator(mode="after")
    def check_candidates(self):
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be at least top_k")
        return self
