from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RagSettings(BaseSettings):
    """Business rules for the RAG flow (all overridable via RAG_* env vars)."""

    score_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum similarity score (0..1, cosine) of the BEST retrieved chunk "
            "required to call the LLM. Below this the canned fallback is returned "
            "and the LLM is never queried."
        ),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="How many chunks to retrieve from ops-core-api per question.",
    )
    max_context_chunks: int = Field(
        default=5,
        ge=1,
        description="Max chunks injected into the LLM prompt as grounding context.",
    )
    message_language: str = Field(
        default="en",
        description=(
            "Language of the canned messages the user receives verbatim (see "
            "src.app.messages). Unknown values fall back to English. This is NOT "
            "the answer language: an LLM answer follows the question's language."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_rag_settings() -> RagSettings:
    return RagSettings()
