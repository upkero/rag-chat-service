import json
from collections.abc import Callable
from dataclasses import replace
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import SecretStr

from src.app.contracts.llm.llm_response import LLMResponse
from src.app.contracts.rag.retrieved_chunk import RetrievedChunk
from src.app.core.settings.rag import RagSettings
from src.app.core.settings.rerank import RerankSettings
from src.app.gateways import factory
from src.app.gateways.factory import create_reranker
from src.app.gateways.http_reranker import HttpReranker
from src.app.services.rag import RagService


def _reranker(handler: Callable[[httpx.Request], httpx.Response]) -> HttpReranker:
    client = httpx.AsyncClient(base_url="https://rerank.test/v1", transport=httpx.MockTransport(handler))
    return HttpReranker(client=client, model="cohere/rerank-v3.5")


def _chunks(chunk: RetrievedChunk, count: int = 3) -> list[RetrievedChunk]:
    return [replace(chunk, chunk_index=index, chunk_text=f"text {index}", score=0.5) for index in range(count)]


async def test_the_request_is_a_cohere_style_rerank_call(cancellation_chunk: RetrievedChunk) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"results": []})

    await _reranker(handler).rerank("how do I cancel", _chunks(cancellation_chunk), top_n=2)

    assert (seen[0].method, str(seen[0].url)) == ("POST", "https://rerank.test/v1/rerank")
    assert json.loads(seen[0].content) == {
        "model": "cohere/rerank-v3.5",
        "query": "how do I cancel",
        "documents": ["text 0", "text 1", "text 2"],
        "top_n": 2,
    }


async def test_top_n_never_exceeds_the_number_of_documents(cancellation_chunk: RetrievedChunk) -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"results": []})

    await _reranker(handler).rerank("q", _chunks(cancellation_chunk, 2), top_n=10)

    assert bodies[0]["top_n"] == 2


async def test_results_keep_the_providers_order_and_replace_the_score(cancellation_chunk: RetrievedChunk) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        results = [{"index": 2, "relevance_score": 0.91}, {"index": 0, "relevance_score": 0.12}]
        return httpx.Response(200, json={"results": results})

    reranked = await _reranker(handler).rerank("q", _chunks(cancellation_chunk), top_n=2)

    assert [(chunk.chunk_index, chunk.score) for chunk in reranked] == [(2, 0.91), (0, 0.12)]
    assert reranked[0].chunk_text == "text 2"


async def test_no_chunks_means_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not be called")

    assert await _reranker(handler).rerank("q", [], top_n=3) == ()


@pytest.mark.parametrize("index", [3, 99, -1])
async def test_an_index_outside_the_documents_is_an_error(cancellation_chunk: RetrievedChunk, index: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": index, "relevance_score": 0.9}]})

    with pytest.raises(ValueError, match="index"):
        await _reranker(handler).rerank("q", _chunks(cancellation_chunk), top_n=1)


async def test_an_http_error_raises(cancellation_chunk: RetrievedChunk) -> None:
    reranker = _reranker(lambda request: httpx.Response(500))

    with pytest.raises(httpx.HTTPStatusError):
        await reranker.rerank("q", _chunks(cancellation_chunk), top_n=1)


@pytest.mark.parametrize(
    "response",
    [httpx.Response(500), httpx.Response(200, json={"results": [{"index": 7, "relevance_score": 0.9}]})],
)
async def test_the_service_falls_back_to_the_embedding_order(
    cancellation_chunk: RetrievedChunk, response: httpx.Response
) -> None:
    gateway, llm = AsyncMock(), AsyncMock()
    gateway.search.return_value = [replace(cancellation_chunk, score=0.82)]
    llm.complete.return_value = LLMResponse(content="Free up to 24 hours before.", usage={})
    service = RagService(
        knowledge=gateway,
        llm_client=llm,
        settings=RagSettings(_env_file=None),
        reranker=_reranker(lambda request: response),
        rerank_settings=RerankSettings(_env_file=None, min_score=0.08),
    )

    answer = await service.ask("how do I cancel")

    assert [source.score for source in answer.sources] == [0.82]  # the embedding score, gated by 0.5


def test_no_reranker_when_it_is_switched_off() -> None:
    assert create_reranker(RerankSettings(_env_file=None, enabled=False)) is None


def test_the_http_reranker_is_wired_with_the_key_and_the_base_url() -> None:
    settings = RerankSettings(
        _env_file=None,
        enabled=True,
        provider="http",
        api_key=SecretStr("rerank-key-0123456789"),
        base_url="https://rerank.test/v1/",
        model="some/model",
    )

    reranker = create_reranker(settings)

    assert isinstance(reranker, HttpReranker)
    assert reranker._model == "some/model"
    assert str(reranker._client.base_url) == "https://rerank.test/v1/"  # httpx adds the slash
    assert reranker._client.headers["Authorization"] == "Bearer rerank-key-0123456789"


def test_the_local_provider_loads_the_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[str] = []

    class FakeLocal:
        def __init__(self, *, model_name: str) -> None:
            loaded.append(model_name)

    monkeypatch.setattr(factory, "LocalReranker", FakeLocal)

    reranker = create_reranker(RerankSettings(_env_file=None, enabled=True, provider="local", local_model="x/y"))

    assert isinstance(reranker, FakeLocal)
    assert loaded == ["x/y"]


async def test_the_rerank_cost_openrouter_reports_is_recorded(
    cancellation_chunk: RetrievedChunk, metric_value: Callable[..., float]
) -> None:
    labels = {"provider": "rerank.test", "model": "cohere/rerank-v3.5"}
    before = metric_value("llm.cost", **labels)

    def handler(request: httpx.Request) -> httpx.Response:
        body = {"results": [{"index": 0, "relevance_score": 0.9}], "usage": {"search_units": 1, "cost": 0.001}}
        return httpx.Response(200, json=body)

    await _reranker(handler).rerank("parking", _chunks(cancellation_chunk), top_n=1)

    assert metric_value("llm.cost", **labels) - before == pytest.approx(0.001)
