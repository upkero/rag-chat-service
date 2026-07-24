import httpx

from src.app.core.settings.ops_core import OpsCoreSettings
from src.app.repositories.knowledge.ops_core_chunk_retriever import OpsCoreChunkRetriever

_API_KEY_HEADER = "X-API-Key"


def create_ops_core_retriever(settings: OpsCoreSettings) -> OpsCoreChunkRetriever:
    """Factory — the only place the ops-core-api HTTP client is constructed.

    Wires settings → a configured httpx.AsyncClient (base URL, auth header,
    timeout) → the retriever adapter, mirroring ``create_llm_client``. Keeping
    construction here lets the container stay declarative and the adapter stay
    purely I/O.
    """
    client = httpx.AsyncClient(
        base_url=settings.base_url.rstrip("/"),
        headers={_API_KEY_HEADER: settings.api_key},
        timeout=settings.timeout_seconds,
    )
    return OpsCoreChunkRetriever(client=client, max_retries=settings.max_retries)
