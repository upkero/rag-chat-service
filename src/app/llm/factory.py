from openai import AsyncOpenAI

from src.app.core.settings.llm import LLMSettings
from src.app.llm.openai_compatible_llm_client import OpenAICompatibleLLMClient


def create_llm_client(settings: LLMSettings) -> OpenAICompatibleLLMClient:
    """Factory — the only place `AsyncOpenAI` is instantiated.

    Wires settings → raw SDK client → our wrapper, so the rest of the app depends
    on the LLMClient interface and never on the SDK directly.
    """
    raw_client = AsyncOpenAI(
        api_key=settings.api_key.get_secret_value() if settings.api_key else None,
        base_url=settings.base_url,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )
    return OpenAICompatibleLLMClient(settings=settings, client=raw_client)
