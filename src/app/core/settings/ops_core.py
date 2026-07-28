from functools import lru_cache

from pydantic import Field, SecretStr
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
    # SecretStr keeps the key out of logs and reprs; the only place it is unwrapped
    # is where the HTTP client is built.
    api_key: SecretStr = Field(
        ...,
        min_length=16,
        description="API key sent as the X-API-Key header to ops-core-api.",
    )
    timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        description="HTTP timeout for ops-core-api calls in seconds.",
    )
    max_attempts: int = Field(
        default=3,
        ge=1,
        description=(
            "TOTAL attempts for an ops-core-api call, including the first one — "
            "not the number of repeats. Every consumer of ops-core-api counts it "
            "this way, so no caller has to remember to add one."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="OPS_CORE_",
        env_file=".env",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_ops_core_settings() -> OpsCoreSettings:
    return OpsCoreSettings()
