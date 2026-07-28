from collections.abc import Sequence
from unittest.mock import AsyncMock

import httpx

from src.app.api.v1.dependencies import get_rag_service
from src.app.api.v1.dependencies.rate_limit import get_rate_limiter
from src.app.contracts.llm.llm_response import LLMResponse
from src.app.contracts.rag.retrieved_chunk import RetrievedChunk
from src.app.core.rate_limit import SlidingWindowRateLimiter
from src.app.core.settings.rag import RagSettings
from src.app.core.settings.security import SecuritySettings, get_security_settings
from src.app.exceptions.llm import LLMGenerationError
from src.app.interfaces.knowledge.chunk_retriever import ChunkRetriever
from src.app.interfaces.llm.llm_client import LLMClient
from src.app.services.rag_service import RagService
from src.main import create_app

TEST_API_KEY = "test-api-key-0123456789"  # >= 16 chars


def _make_service(
    *,
    chunks: Sequence[RetrievedChunk],
    llm_content: str = "unused",
    llm_error: Exception | None = None,
) -> RagService:
    retriever = AsyncMock(spec=ChunkRetriever)
    retriever.search.return_value = list(chunks)
    llm = AsyncMock(spec=LLMClient)
    if llm_error is not None:
        llm.complete.side_effect = llm_error
    else:
        llm.complete.return_value = LLMResponse(content=llm_content, usage={"total_tokens": 42})
    return RagService(retriever=retriever, llm_client=llm, settings=RagSettings())


def _app_for(service: RagService, *, rate_limit_max: int = 1000) -> httpx.ASGITransport:
    app = create_app()
    # dependency_overrides is the test seam: mock the service, pin the API key,
    # and use one shared (isolated) rate limiter — no real ops-core-api / LLM
    # touched. The limiter is built ONCE here (not inside the lambda) so its
    # window state persists across requests within the test.
    limiter = SlidingWindowRateLimiter(max_requests=rate_limit_max, window_seconds=60.0)
    app.dependency_overrides[get_rag_service] = lambda: service
    app.dependency_overrides[get_security_settings] = lambda: SecuritySettings(api_key=TEST_API_KEY)
    app.dependency_overrides[get_rate_limiter] = lambda: limiter
    return httpx.ASGITransport(app=app)


def _client_for(service: RagService, *, rate_limit_max: int = 1000) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=_app_for(service, rate_limit_max=rate_limit_max),
        base_url="http://test",
        headers={"X-API-Key": TEST_API_KEY},
    )


async def test_ask_confident_returns_answer_and_sources() -> None:
    chunk = RetrievedChunk(
        document_title="Прайс-лист",
        chunk_index=2,
        chunk_text="Мезотерапия лица — 3500 рублей.",
        score=0.82,
    )
    service = _make_service(chunks=[chunk], llm_content="Мезотерапия лица стоит 3500 рублей.")

    async with _client_for(service) as client:
        response = await client.post("/api/v1/rag-chat/ask", json={"question": "сколько стоит мезотерапия"})

    assert response.status_code == 200
    body = response.json()
    assert "3500" in body["answer"]
    assert body["sources"] == [
        {"document_title": "Прайс-лист", "chunk_index": 2, "score": 0.82},
    ]


async def test_ask_below_threshold_returns_fallback_without_sources() -> None:
    chunk = RetrievedChunk(document_title="FAQ", chunk_index=0, chunk_text="...", score=0.2)
    service = _make_service(chunks=[chunk])

    async with _client_for(service) as client:
        response = await client.post("/api/v1/rag-chat/ask", json={"question": "какая погода на марсе"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == RagSettings().fallback_message
    assert body["sources"] == []


async def test_ask_generation_failure_returns_502_envelope() -> None:
    chunk = RetrievedChunk(
        document_title="Прайс-лист",
        chunk_index=1,
        chunk_text="Мезотерапия лица — 3500 рублей.",
        score=0.9,
    )
    service = _make_service(chunks=[chunk], llm_error=LLMGenerationError("provider down"))

    async with _client_for(service) as client:
        response = await client.post("/api/v1/rag-chat/ask", json={"question": "сколько стоит мезотерапия"})

    assert response.status_code == 502
    assert response.json()["error_code"] == "generation_unavailable"


async def test_ask_rejects_empty_question() -> None:
    service = _make_service(chunks=[])

    async with _client_for(service) as client:
        response = await client.post("/api/v1/rag-chat/ask", json={"question": ""})

    assert response.status_code == 422


async def test_ask_without_api_key_returns_401() -> None:
    service = _make_service(chunks=[])
    # A client that deliberately sends no X-API-Key header.
    async with httpx.AsyncClient(transport=_app_for(service), base_url="http://test") as client:
        response = await client.post("/api/v1/rag-chat/ask", json={"question": "hi"})

    assert response.status_code == 401
    assert response.json()["error_code"] == "unauthorized"


async def test_ask_with_wrong_api_key_returns_401() -> None:
    service = _make_service(chunks=[])
    async with httpx.AsyncClient(transport=_app_for(service), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/rag-chat/ask",
            json={"question": "hi"},
            headers={"X-API-Key": "wrong-key-0123456789"},
        )

    assert response.status_code == 401


async def test_ask_rate_limited_after_limit() -> None:
    chunk = RetrievedChunk(
        document_title="Прайс-лист",
        chunk_index=0,
        chunk_text="Мезотерапия лица — 3500 рублей.",
        score=0.9,
    )
    service = _make_service(chunks=[chunk], llm_content="ответ")

    async with _client_for(service, rate_limit_max=1) as client:
        first = await client.post("/api/v1/rag-chat/ask", json={"question": "вопрос"})
        second = await client.post("/api/v1/rag-chat/ask", json={"question": "вопрос"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error_code"] == "rate_limit_exceeded"
