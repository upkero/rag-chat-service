from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RateLimitSettings(BaseSettings):
    """Per-client throttle for the ask endpoint (protects LLM spend / abuse)."""

    # 20 is ops-core-api's EMBEDDING_RATE_LIMIT_PER_MINUTE, and every question here
    # costs exactly one call to that endpoint. The rule this follows: your own limit
    # never exceeds the limit of the upstream you call on every request. Letting 60
    # through into a budget of 20 does not serve more users, it just moves the 429
    # one hop further away from the client that has to act on it.
    max_requests: int = Field(
        default=20,
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
