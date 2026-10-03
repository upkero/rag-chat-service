from collections.abc import Sequence
from unittest.mock import AsyncMock
from uuid import UUID

import httpx
from starlette.middleware.cors import CORSMiddleware

from src.app.api.v1.dependencies import get_rag_service
from src.app.api.v1.dependencies.rate_limit import get_rate_limiter
from src.app.contracts.llm.llm_response import LLMResponse
from src.app.contracts.rag.retrieved_chunk import RetrievedChunk
from src.app.core.rate_limit import SlidingWindowRateLimiter
from src.app.core.settings.rag import RagSettings
from src.app.core.settings.security import SecuritySettings, get_security_settings
from src.app.exceptions.llm import LLMGenerationError
from src.app.interfaces.knowledge.knowledge_gateway import KnowledgeGateway
from src.app.interfaces.llm.llm_client import LLMClient
from src.app.messages import get_message
from src.app.services.rag import RagService
from src.main import create_app

TEST_API_KEY = "test-api-key-0123456789"  # >= 16 chars


def _seed_chunk(*, chunk_index: int, score: float) -> RetrievedChunk:
    # Real content from ops-core-api's seed data, so the examples these tests
    # encode are the same ones the README tells a reader to curl.
    return RetrievedChunk(
        document_title="Cancellation and rescheduling policy",
        chunk_index=chunk_index,
        chunk_text=(
            "Appointments can be cancelled or rescheduled free of charge up to "
            "twenty-four hours before the scheduled start time."
        ),
        score=score,
    )


def _make_service(
    *,
    chunks: Sequence[RetrievedChunk],
    llm_content: str = "unused",
    llm_error: Exception | None = None,
) -> RagService:
    gateway = AsyncMock(spec=KnowledgeGateway)
    gateway.search.return_value = list(chunks)
    llm = AsyncMock(spec=LLMClient)
    if llm_error is not None:
        llm.complete.side_effect = llm_error
    else:
        llm.complete.return_value = LLMResponse(content=llm_content, usage={"total_tokens": 42})
    return RagService(knowledge=gateway, llm_client=llm, settings=RagSettings(_env_file=None))


def _app_for(service: RagService, *, rate_limit_max: int = 1000) -> httpx.ASGITransport:
    app = create_app()
    # dependency_overrides is the test seam: mock the service, pin the API key,
    # and use one shared (isolated) rate limiter — no real ops-core-api / LLM
    # touched. The limiter is built ONCE here (not inside the lambda) so its
    # window state persists across requests within the test.
    limiter = SlidingWindowRateLimiter(max_requests=rate_limit_max, window_seconds=60.0)
    app.dependency_overrides[get_rag_service] = lambda: service
    app.dependency_overrides[get_security_settings] = lambda: SecuritySettings(_env_file=None, api_key=TEST_API_KEY)
    app.dependency_overrides[get_rate_limiter] = lambda: limiter
    return httpx.ASGITransport(app=app)


def _client_for(service: RagService, *, rate_limit_max: int = 1000) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=_app_for(service, rate_limit_max=rate_limit_max),
        base_url="http://test",
        headers={"X-API-Key": TEST_API_KEY},
    )


async def test_ask_confident_returns_answer_and_sources() -> None:
    chunk = _seed_chunk(chunk_index=2, score=0.82)
    service = _make_service(
        chunks=[chunk],
        llm_content="You can cancel free of charge up to twenty-four hours before the start time.",
    )

    async with _client_for(service) as client:
        response = await client.post("/api/v1/ask", json={"question": "how do I cancel an appointment"})

    assert response.status_code == 200
    body = response.json()
    assert "twenty-four hours" in body["answer"]
    assert body["sources"] == [
        {"document_title": "Cancellation and rescheduling policy", "chunk_index": 2, "score": 0.82},
    ]


async def test_ask_below_threshold_returns_fallback_without_sources() -> None:
    chunk = _seed_chunk(chunk_index=0, score=0.2)
    service = _make_service(chunks=[chunk])

    async with _client_for(service) as client:
        response = await client.post("/api/v1/ask", json={"question": "what is the weather on mars"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == get_message("low_confidence", "en")
    assert body["sources"] == []


async def test_ask_generation_failure_returns_502_envelope() -> None:
    chunk = _seed_chunk(chunk_index=1, score=0.9)
    service = _make_service(chunks=[chunk], llm_error=LLMGenerationError("provider down"))

    async with _client_for(service) as client:
        response = await client.post("/api/v1/ask", json={"question": "how do I cancel an appointment"})

    assert response.status_code == 502
    assert response.json()["error_code"] == "generation_unavailable"


async def test_ask_rejects_empty_question() -> None:
    service = _make_service(chunks=[])

    async with _client_for(service) as client:
        response = await client.post("/api/v1/ask", json={"question": ""})

    assert response.status_code == 422


async def test_ask_rejects_whitespace_only_question_without_searching() -> None:
    service = _make_service(chunks=[])

    async with _client_for(service) as client:
        for question in ("   ", "\n\t "):
            response = await client.post("/api/v1/ask", json={"question": question})

            assert response.status_code == 422
            assert response.json()["error_code"] == "request_validation_error"

    service._knowledge.search.assert_not_awaited()  # type: ignore[attr-defined]


async def test_ask_without_api_key_returns_401() -> None:
    service = _make_service(chunks=[])
    # A client that deliberately sends no X-API-Key header.
    async with httpx.AsyncClient(transport=_app_for(service), base_url="http://test") as client:
        response = await client.post("/api/v1/ask", json={"question": "hi"})

    assert response.status_code == 401
    assert response.json()["error_code"] == "invalid_api_key"


async def test_ask_with_wrong_api_key_returns_401() -> None:
    service = _make_service(chunks=[])
    async with httpx.AsyncClient(transport=_app_for(service), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/ask",
            json={"question": "hi"},
            headers={"X-API-Key": "wrong-key-0123456789"},
        )

    assert response.status_code == 401


async def test_ask_rate_limited_after_limit() -> None:
    chunk = _seed_chunk(chunk_index=0, score=0.9)
    service = _make_service(chunks=[chunk], llm_content="an answer")

    async with _client_for(service, rate_limit_max=1) as client:
        first = await client.post("/api/v1/ask", json={"question": "a question"})
        second = await client.post("/api/v1/ask", json={"question": "a question"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error_code"] == "rate_limit_exceeded"
    assert 1 <= int(second.headers["Retry-After"]) <= 60


def test_cors_exposes_the_headers_a_browser_needs() -> None:
    app = create_app()
    cors = next(m for m in app.user_middleware if m.cls is CORSMiddleware)

    assert set(cors.kwargs["expose_headers"]) == {"Retry-After", "X-Request-ID"}


async def test_ask_form_body_is_a_validation_error_not_a_crash() -> None:
    # curl's default content type when -H 'Content-Type: application/json' is
    # forgotten. Pydantic keeps the raw bytes in the error, undecodable ones too.
    async with _client_for(_make_service(chunks=[])) as client:
        for body in (b"question=hi", b"\xff\xfe"):
            response = await client.post(
                "/api/v1/ask",
                content=body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            assert response.status_code == 422
            assert response.json()["error_code"] == "request_validation_error"


async def test_unhandled_error_is_a_500_that_still_carries_the_request_id() -> None:
    async def boom() -> None:
        raise RuntimeError("boom")

    app = create_app()
    app.add_api_route("/boom", boom)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/boom", headers={"X-Request-ID": "abc-123"})

    assert response.status_code == 500
    assert response.json()["error_code"] == "internal_server_error"
    assert response.headers["X-Request-ID"] == "abc-123"


def test_cors_is_the_outermost_middleware() -> None:
    # Starlette keeps the outermost first. Otherwise a 429 reaches the browser without
    # CORS headers and an OPTIONS preflight spends the request id and rate budget.
    assert create_app().user_middleware[0].cls is CORSMiddleware


async def test_malformed_request_id_is_replaced() -> None:
    async with _client_for(_make_service(chunks=[])) as client:
        # Echoed, forwarded upstream and logged, so a markup or oversized id is replaced.
        for bad in ("attacker-<script>", "r" * 129):
            response = await client.get("/no-such-route", headers={"X-Request-ID": bad})

            assert UUID(response.headers["X-Request-ID"])
