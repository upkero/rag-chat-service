import pytest
from pydantic import ValidationError

from src.app.core.settings.llm import LLMSettings
from src.app.core.settings.security import SecuritySettings

_SECRET = "s3cret-value-0123456789"


def test_api_keys_never_show_in_a_repr_or_a_dump() -> None:
    security = SecuritySettings(_env_file=None, api_key=_SECRET)
    llm = LLMSettings(_env_file=None, provider="openai", model="m", api_key=_SECRET)

    for settings in (security, llm):
        assert _SECRET not in repr(settings)
        assert _SECRET not in str(settings.model_dump())
    assert security.api_key.get_secret_value() == _SECRET


def test_a_too_short_security_key_does_not_echo_itself_in_the_error() -> None:
    with pytest.raises(ValidationError) as error:
        SecuritySettings(_env_file=None, api_key="short")

    assert "input_value" not in str(error.value)


def test_the_openai_provider_still_demands_a_key() -> None:
    with pytest.raises(ValueError, match="api_key is required"):
        LLMSettings(_env_file=None, provider="openai", model="m")
