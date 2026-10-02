from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RerankSettings(BaseSettings):
    """Optional cross-encoder rerank step (all overridable via RERANK_* env vars)."""

    enabled: bool = Field(default=False, description="Rerank retrieved chunks before the confidence gate.")
    provider: Literal["http", "local"] = Field(
        default="http",
        description="'http' calls a rerank API (OpenRouter by default); 'local' runs a sentence-transformers CrossEncoder in-process.",
    )
    local_model: str = Field(default="BAAI/bge-reranker-v2-m3", description="CrossEncoder used when provider='local'.")
    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Server exposing a Cohere-style POST {base_url}/rerank.",
    )
    api_key: SecretStr | None = Field(default=None, description="Required when enabled with provider='http'.")
    model: str = Field(default="cohere/rerank-v3.5", description="Cross-encoder model name.")
    timeout_seconds: float = Field(default=10.0, gt=0)
    candidates: int = Field(
        default=10, ge=1, le=50, description="How many chunks to retrieve and hand to the reranker."
    )
    min_score: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description=(
            "Gate for the BEST reranked chunk, replacing RAG_SCORE_THRESHOLD while reranking "
            "is active. Rerank scores are not cosine similarities: relevant chunks "
            "scored 0.1-0.7 and off-topic ones <= 0.02 against Cohere rerank-v3.5; local models differ."
        ),
    )

    model_config = SettingsConfigDict(env_prefix="RERANK_", env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def _key_when_enabled(self) -> "RerankSettings":
        if self.enabled and self.provider == "http" and self.api_key is None:
            raise ValueError("RERANK_API_KEY is required when RERANK_ENABLED=true and RERANK_PROVIDER=http.")
        return self


@lru_cache(maxsize=1)
def get_rerank_settings() -> RerankSettings:
    return RerankSettings()
