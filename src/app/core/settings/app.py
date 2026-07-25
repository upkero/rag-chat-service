from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Root application settings — HTTP-surface concerns only.

    Unlike the shared template this service carries NO database configuration:
    it stores nothing itself and delegates all retrieval to ops-core-api.
    """

    cors_allow_origins: list[str] = Field(
        default=[],
        description=(
            "Allowed CORS origins. Empty by default (no cross-origin access) — "
            "set explicitly per environment. Avoid ['*'] in production."
        ),
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_app_settings() -> AppSettings:
    return AppSettings()
