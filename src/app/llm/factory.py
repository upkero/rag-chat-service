from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

from src.app.core.settings.llm import LLMSettings
from src.app.llm.openai_compatible_llm_client import OpenAICompatibleLLMClient


def create_llm_client(settings: LLMSettings) -> OpenAICompatibleLLMClient:
    """Factory — the only place `AsyncOpenAI` is instantiated.

    Wires settings → raw SDK client → our wrapper, so the rest of the app depends
    on the LLMClient interface and never on the SDK directly.
    """
    # The SDK builds its own httpx client; handing it one lets each attempt, SDK retries
    # included, be a client span and carry a `traceparent`. The SDK still applies the
    # timeout and retry settings per request.
    http_client = DefaultAsyncHttpxClient()
    HTTPXClientInstrumentor().instrument_client(http_client)
    raw_client = AsyncOpenAI(
        http_client=http_client,
        api_key=settings.api_key.get_secret_value() if settings.api_key else None,
        base_url=settings.base_url,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )
    return OpenAICompatibleLLMClient(settings=settings, client=raw_client)
