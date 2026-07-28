from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SecuritySettings(BaseSettings):
    """Auth settings for this service's own HTTP surface.

    Callers must present ``api_key`` in the ``X-API-Key`` header. Required (min
    length 16) so the service is never accidentally deployed wide open — the same
    posture ops-core-api uses for its endpoints.
    """

    api_key: str = Field(
        ...,
        min_length=16,
        description="API key required in the X-API-Key header to call /api/v1/*.",
    )

    model_config = SettingsConfigDict(
        env_prefix="SECURITY_",
        env_file=".env",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_security_settings() -> SecuritySettings:
    return SecuritySettings()
