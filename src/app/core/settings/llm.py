from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["openai", "openai_compatible", "ollama"]


class LLMSettings(BaseSettings):
    provider: LLMProvider = Field(
        default="openai",
        description=(
            "LLM provider profile. Ollama and openai_compatible use the "
            "OpenAI-compatible client with a provider-specific base URL."
        ),
    )
    model: str = Field(
        ...,
        min_length=1,
        description="LLM model name sent to the provider.",
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="Provider API key. Required for provider='openai'.",
    )
    base_url: str | None = Field(
        default=None,
        description="Optional OpenAI-compatible API base URL.",
    )
    timeout_seconds: float = Field(
        default=15.0,
        gt=0,
        description=(
            "LLM client timeout in seconds, per attempt. The worst case is this times "
            "(LLM_MAX_RETRIES + 1), and it has to leave room under the 90 s the site's bridge waits, "
            "next to retrieval (10 s per attempt) and the rerank call (10 s): 15 s x 3 = 45 s."
        ),
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        description="Maximum provider SDK retries, on top of the first attempt.",
    )
    ping_ok_ttl_seconds: float = Field(
        default=30.0,
        ge=0.0,
        description=(
            "How long a healthy readiness probe of the provider is reused. Readiness is polled often "
            "and each probe is a real request, so this trades detection lag for load. 0 disables."
        ),
    )
    ping_failed_ttl_seconds: float = Field(
        default=5.0,
        ge=0.0,
        description="How long a failed readiness probe is reused. Short, so a recovery shows on the next poll.",
    )
    temperature: float | None = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Set to None to omit it.",
    )
    max_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Optional maximum number of output tokens.",
    )
    reasoning_effort: str | None = Field(
        default=None,
        description="Reasoning effort level (o-series / xAI models).",
    )
    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        extra="ignore",
        hide_input_in_errors=True,
    )

    @model_validator(mode="after")
    def validate_provider_requirements(self) -> "LLMSettings":
        if self.provider == "openai" and not self.api_key:
            raise ValueError("api_key is required when provider='openai'.")
        if self.provider != "openai" and not self.base_url:
            raise ValueError(
                "base_url is required when provider is openai_compatible or ollama."
            )
        return self


@lru_cache(maxsize=1)
def get_llm_settings() -> LLMSettings:
    return LLMSettings()
