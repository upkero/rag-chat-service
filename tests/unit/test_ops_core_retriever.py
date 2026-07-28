import json
from collections.abc import Callable

import httpx
import pytest

from src.app.core.settings.ops_core import OpsCoreSettings
from src.app.exceptions.retrieval import RetrievalUnavailableError
from src.app.repositories.knowledge.factory import create_ops_core_retriever
from src.app.repositories.knowledge.ops_core_chunk_retriever import OpsCoreChunkRetriever

Handler = Callable[[httpx.Request], httpx.Response]


def _retriever(handler: Handler, *, max_retries: int = 2) -> OpsCoreChunkRetriever:
    # MockTransport lets us drive the adapter without a real ops-core-api.
    # retry_base_delay=0 keeps the retry tests instant.
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ops-core")
    return OpsCoreChunkRetriever(client=client, max_retries=max_retries, retry_base_delay=0.0)


async def test_search_maps_matches_to_chunks_and_sends_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/documents/search"
        assert json.loads(request.content) == {"query": "мезотерапия", "top_k": 3}
        return httpx.Response(
            200,
            json={
                "query": "мезотерапия",
                "matches": [
                    {
                        "chunk_id": "c1",
                        "document_id": "d1",
                        "document_title": "Прайс-лист",
                        "chunk_index": 2,
                        "chunk_text": "Мезотерапия — 3500 руб.",
                        "distance": 0.1,
                        "score": 0.9,
                    }
                ],
            },
        )

    retriever = _retriever(handler)
    chunks = await retriever.search("мезотерапия", 3)
    await retriever.close()

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.document_title == "Прайс-лист"
    assert chunk.chunk_index == 2
    assert chunk.chunk_text == "Мезотерапия — 3500 руб."
    assert chunk.score == 0.9


async def test_search_4xx_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"detail": "bad key"})

    retriever = _retriever(handler, max_retries=2)
    with pytest.raises(RetrievalUnavailableError):
        await retriever.search("q", 5)
    await retriever.close()

    assert calls["n"] == 1  # a client error must not be retried


async def test_search_5xx_is_retried_then_raises() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503)

    retriever = _retriever(handler, max_retries=2)  # 1 + 2 retries = 3 attempts
    with pytest.raises(RetrievalUnavailableError):
        await retriever.search("q", 5)
    await retriever.close()

    assert calls["n"] == 3


async def test_search_malformed_schema_raises_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"matches": [{"document_title": "x"}]})  # missing fields

    retriever = _retriever(handler)
    with pytest.raises(RetrievalUnavailableError):
        await retriever.search("q", 5)
    await retriever.close()


async def test_ping_reflects_backend_health() -> None:
    async def ping_with_status(status: int) -> bool:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/health/ready"
            return httpx.Response(status)

        retriever = _retriever(handler)
        result = await retriever.ping()
        await retriever.close()
        return result

    assert await ping_with_status(200) is True
    # 503 is what ops-core-api returns when its own database is unreachable —
    # the case the old `< 500` check reported as healthy.
    assert await ping_with_status(503) is False
    assert await ping_with_status(404) is False


async def test_factory_configures_auth_header_and_base_url() -> None:
    settings = OpsCoreSettings(
        base_url="http://ops-core:8000/",
        api_key="k" * 16,
        timeout_seconds=5.0,
        max_retries=1,
    )
    retriever = create_ops_core_retriever(settings)

    client = retriever._client
    assert client.headers["X-API-Key"] == "k" * 16
    assert client.base_url.host == "ops-core"
    await retriever.close()
