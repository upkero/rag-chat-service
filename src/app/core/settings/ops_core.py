from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpsCoreSettings(BaseSettings):
    """Connection settings for the upstream ops-core-api document search.

    Connection concerns only — how many chunks to retrieve is a RAG business
    decision and lives in RagSettings instead.
    """

    base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL of ops-core-api (WITHOUT the /api/v1 suffix).",
    )
    api_key: str = Field(
        ...,
        min_length=16,
        description="API key sent as the X-API-Key header to ops-core-api.",
    )
    timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        description="HTTP timeout for ops-core-api calls in seconds.",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        description="Retry attempts for transient ops-core-api failures.",
    )

    model_config = SettingsConfigDict(
        env_prefix="OPS_CORE_",
        env_file=".env",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_ops_core_settings() -> OpsCoreSettings:
    # Required fields are populated from the environment; mypy can't see that.
    return OpsCoreSettings()  # type: ignore[call-arg]
