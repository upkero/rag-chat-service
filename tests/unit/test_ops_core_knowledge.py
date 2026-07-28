import json
from collections.abc import Callable

import httpx
import pytest
from pydantic import SecretStr

from src.app.core.settings.ops_core import OpsCoreSettings
from src.app.exceptions.retrieval import RetrievalThrottledError, RetrievalUnavailableError
from src.app.gateways.factory import create_ops_core_knowledge_gateway
from src.app.gateways.ops_core_knowledge import OpsCoreKnowledgeGateway

Handler = Callable[[httpx.Request], httpx.Response]


def _gateway(handler: Handler, *, max_attempts: int = 3) -> OpsCoreKnowledgeGateway:
    # MockTransport lets us drive the adapter without a real ops-core-api.
    # max_wait_seconds=0 keeps the retry tests instant.
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ops-core")
    return OpsCoreKnowledgeGateway(client=client, max_attempts=max_attempts, max_wait_seconds=0.0)


async def test_search_maps_matches_to_chunks_and_sends_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/documents/search"
        assert json.loads(request.content) == {"query": "how do I cancel", "top_k": 3}
        return httpx.Response(
            200,
            json={
                "query": "how do I cancel",
                "matches": [
                    {
                        "chunk_id": "c1",
                        "document_id": "d1",
                        "document_title": "Cancellation and rescheduling policy",
                        "chunk_index": 2,
                        "chunk_text": "Cancel free of charge up to twenty-four hours before the start time.",
                        "distance": 0.1,
                        "score": 0.9,
                    }
                ],
            },
        )

    gateway = _gateway(handler)
    chunks = await gateway.search("how do I cancel", 3)
    await gateway.close()

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.document_title == "Cancellation and rescheduling policy"
    assert chunk.chunk_index == 2
    assert chunk.chunk_text == "Cancel free of charge up to twenty-four hours before the start time."
    assert chunk.score == 0.9


async def test_search_4xx_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"detail": "bad key"})

    gateway = _gateway(handler, max_attempts=3)
    with pytest.raises(RetrievalUnavailableError):
        await gateway.search("q", 5)
    await gateway.close()

    assert calls["n"] == 1  # a client error must not be retried


async def test_search_5xx_is_retried_then_raises() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503)

    gateway = _gateway(handler, max_attempts=3)
    with pytest.raises(RetrievalUnavailableError):
        await gateway.search("q", 5)
    await gateway.close()

    assert calls["n"] == 3


async def test_search_malformed_schema_raises_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"matches": [{"document_title": "x"}]})  # missing fields

    gateway = _gateway(handler)
    with pytest.raises(RetrievalUnavailableError):
        await gateway.search("q", 5)
    await gateway.close()


async def test_ping_reflects_backend_health() -> None:
    async def ping_with_status(status: int) -> bool:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/health/ready"
            return httpx.Response(status)

        gateway = _gateway(handler)
        result = await gateway.ping()
        await gateway.close()
        return result

    assert await ping_with_status(200) is True
    # 503 is what ops-core-api returns when its own database is unreachable —
    # the case the old `< 500` check reported as healthy.
    assert await ping_with_status(503) is False
    assert await ping_with_status(404) is False


async def test_factory_configures_auth_header_and_base_url() -> None:
    settings = OpsCoreSettings(
        base_url="http://ops-core:8000/",
        api_key=SecretStr("k" * 16),
        timeout_seconds=5.0,
        max_attempts=2,
    )
    gateway = create_ops_core_knowledge_gateway(settings)

    client = gateway._client
    assert client.headers["X-API-Key"] == "k" * 16
    assert client.base_url.host == "ops-core"
    await gateway.close()


async def test_search_retries_a_throttled_upstream_then_reports_429() -> None:
    """A 429 that survives every attempt must stay a 429, carrying Retry-After.

    Turning "the upstream asked you to wait 7 seconds" into a 502 "service is
    unavailable" throws away the only actionable thing the caller was told, and
    the caller cannot tell a throttle from an outage.
    """
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        # Retry-After is honoured as a real sleep, so the attempts we intend to
        # burn through say 0. Only the last one carries the value we assert on,
        # which keeps the test both precise and instant.
        retry_after = "7" if calls == 3 else "0"
        return httpx.Response(429, headers={"Retry-After": retry_after}, json={"detail": "slow down"})

    with pytest.raises(RetrievalThrottledError) as excinfo:
        await _gateway(handler, max_attempts=3).search("how do I cancel", 3)

    assert calls == 3, "429 must be retried, not surfaced on the first response"
    assert excinfo.value.status_code == 429
    assert excinfo.value.headers["Retry-After"] == "7"


async def test_search_recovers_when_a_throttle_clears_within_the_budget() -> None:
    """The retry is worth doing: a 429 followed by a 200 returns chunks, not an error."""
    responses = [
        httpx.Response(429, headers={"Retry-After": "0"}, json={"detail": "slow down"}),
        httpx.Response(200, json={"query": "q", "matches": []}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    assert await _gateway(handler, max_attempts=3).search("q", 3) == []
