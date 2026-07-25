from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RateLimitSettings(BaseSettings):
    """Per-client throttle for the ask endpoint (protects LLM spend / abuse)."""

    max_requests: int = Field(
        default=60,
        ge=1,
        description="Max requests allowed per client within the window.",
    )
    window_seconds: float = Field(
        default=60.0,
        gt=0,
        description="Sliding window length in seconds.",
    )

    model_config = SettingsConfigDict(
        env_prefix="RATE_LIMIT_",
        env_file=".env",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_rate_limit_settings() -> RateLimitSettings:
    return RateLimitSettings()
