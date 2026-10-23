"""What an operator can read off the spans and metrics, and what must never be on them.

The LLM client is the real one over a fake SDK, so the `chat` span, its attributes and
its metrics are the production code's; nothing here asserts on a mock's own calls.
"""

import logging
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.app.api.v1.dependencies import get_rag_service
from src.app.api.v1.dependencies.rate_limit import get_rate_limiter
from src.app.contracts.rag.retrieved_chunk import RetrievedChunk
from src.app.core.rate_limit import SlidingWindowRateLimiter
from src.app.core.resilience import retry_async
from src.app.core.settings.llm import LLMSettings
from src.app.core.settings.rag import RagSettings
from src.app.core.settings.rerank import RerankSettings
from src.app.core.settings.security import SecuritySettings, get_security_settings
from src.app.core.telemetry import UNMATCHED_ROUTE
from src.app.exceptions.llm import GenerationUnavailableError
from src.app.exceptions.retrieval import RetrievalUnavailableError
from src.app.interfaces.knowledge.knowledge_gateway import KnowledgeGateway
from src.app.interfaces.rerank.reranker import Reranker
from src.app.llm.openai_compatible_llm_client import OpenAICompatibleLLMClient
from src.app.services.rag import RagService
from src.main import create_app

QUESTION = "how do I cancel my appointment with Dr Zorbak"
OPENROUTER = "https://openrouter.ai/api/v1"
MODEL = "openai/gpt-4o-mini"
API_KEY = "test-api-key-0123456789"

MetricValue = Callable[..., float]


def _completion(*, cost: float | None = 0.0004) -> SimpleNamespace:
    usage = SimpleNamespace(prompt_tokens=120, completion_tokens=17, total_tokens=137)
    if cost is not None:
        usage.cost = cost
    choice = SimpleNamespace(message=SimpleNamespace(content="Cancel up to 24 hours before."), finish_reason="stop")
    return SimpleNamespace(id="gen-1", created=1, model=MODEL, choices=[choice], usage=usage)


def _llm(base_url: str = OPENROUTER, completion: object | None = None) -> tuple[OpenAICompatibleLLMClient, MagicMock]:
    sdk = MagicMock()
    sdk.chat.completions.create = AsyncMock(return_value=completion or _completion())
    settings = LLMSettings(_env_file=None, provider="openai_compatible", model=MODEL, base_url=base_url)
    return OpenAICompatibleLLMClient(settings=settings, client=sdk), sdk


def _service(
    llm: OpenAICompatibleLLMClient,
    chunk: RetrievedChunk,
    *,
    reranker: Reranker | None = None,
) -> RagService:
    gateway = AsyncMock(spec=KnowledgeGateway)
    gateway.search.return_value = [chunk]
    return RagService(
        knowledge=gateway,
        llm_client=llm,
        settings=RagSettings(_env_file=None, score_threshold=0.5, top_k=5, max_context_chunks=5),
        reranker=reranker,
        rerank_settings=RerankSettings(_env_file=None, candidates=10, min_score=0.1) if reranker else None,
    )


def _span(spans: InMemorySpanExporter, name: str) -> ReadableSpan:
    return next(span for span in spans.get_finished_spans() if span.name == name)


def _attrs(spans: InMemorySpanExporter, name: str) -> dict[str, Any]:
    return dict(_span(spans, name).attributes or {})


async def test_the_chat_span_has_token_usage_and_nothing_from_the_question(
    spans: InMemorySpanExporter, cancellation_chunk: RetrievedChunk
) -> None:
    llm, _ = _llm()

    await _service(llm, cancellation_chunk).ask(QUESTION)

    chat = _span(spans, f"chat {MODEL}")
    attributes = _attrs(spans, f"chat {MODEL}")
    assert chat.kind == trace.SpanKind.CLIENT
    assert attributes["gen_ai.usage.input_tokens"] == 120
    assert attributes["gen_ai.usage.output_tokens"] == 17
    assert attributes["gen_ai.provider.name"] == "openrouter"
    assert attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert chat.parent is not None
    assert chat.parent.span_id == _span(spans, "rag.ask").context.span_id
    # Not the question, not the reply, not the chunk text: on any span, event or attribute.
    forbidden = (QUESTION, "Zorbak", "Cancel up to 24 hours", "twenty-four hours")
    for span in spans.get_finished_spans():
        values = [str(value) for value in (span.attributes or {}).values()]
        values += [str(value) for event in span.events for value in (event.attributes or {}).values()]
        assert not [v for v in values if any(word in v for word in forbidden)], span.name


async def test_rag_ask_records_how_the_answer_came_about(
    spans: InMemorySpanExporter, metric_value: MetricValue, cancellation_chunk: RetrievedChunk
) -> None:
    llm, _ = _llm()
    before = metric_value("rag.answers", outcome="answered", reranked=False)

    await _service(llm, cancellation_chunk).ask(QUESTION)

    assert metric_value("rag.answers", outcome="answered", reranked=False) - before == 1
    ask = _attrs(spans, "rag.ask")
    assert ask["rag.outcome"] == "answered"
    assert ask["rag.reranked"] is False
    assert ask["rag.chunks.retrieved"] == 1
    assert ask["rag.chunks.in_context"] == 1
    assert ask["rag.best_score"] == 0.82
    assert ask["question.length"] == len(QUESTION)
    assert str(ask["rag.prompt.id"]).startswith("rag.answer.system@")
    assert str(ask["rag.user_prompt.id"]).startswith("rag.answer.user@")
    retrieve = _span(spans, "rag.retrieve")
    assert retrieve.parent is not None
    assert retrieve.parent.span_id == _span(spans, "rag.ask").context.span_id


async def test_a_weak_retrieval_is_counted_as_the_gate_and_never_calls_the_model(
    metric_value: MetricValue, cancellation_chunk: RetrievedChunk
) -> None:
    llm, sdk = _llm()
    weak = RetrievedChunk(document_title="t", chunk_index=0, chunk_text="x", score=0.1)
    before = metric_value("rag.answers", outcome="low_confidence_gate")

    await _service(llm, weak).ask(QUESTION)

    assert metric_value("rag.answers", outcome="low_confidence_gate") - before == 1
    sdk.chat.completions.create.assert_not_awaited()


async def test_a_failed_retrieval_is_counted_as_an_error_and_marks_the_span(
    spans: InMemorySpanExporter, metric_value: MetricValue, cancellation_chunk: RetrievedChunk
) -> None:
    llm, _ = _llm()
    service = _service(llm, cancellation_chunk)
    service._knowledge.search.side_effect = RetrievalUnavailableError("down")  # type: ignore[attr-defined]
    before = metric_value("rag.answers", outcome="error")

    with pytest.raises(RetrievalUnavailableError):
        await service.ask(QUESTION)

    assert metric_value("rag.answers", outcome="error") - before == 1
    assert _span(spans, "rag.ask").status.status_code == trace.StatusCode.ERROR


async def test_a_failing_reranker_is_a_counted_fallback_not_a_silent_one(
    spans: InMemorySpanExporter, metric_value: MetricValue, cancellation_chunk: RetrievedChunk
) -> None:
    llm, _ = _llm()
    reranker = AsyncMock(spec=Reranker)
    reranker.rerank.side_effect = httpx.ConnectError("boom")
    before = metric_value("rag.rerank.fallbacks")

    await _service(llm, cancellation_chunk, reranker=reranker).ask(QUESTION)

    assert metric_value("rag.rerank.fallbacks") - before == 1
    rerank = _attrs(spans, "rag.rerank")
    assert rerank["rerank.fallback"] is True
    assert rerank["error.type"] == "ConnectError"
    assert _attrs(spans, "rag.ask")["rag.reranked"] is False


async def test_openrouter_is_asked_for_the_exact_cost_and_it_is_recorded(
    metric_value: MetricValue, cancellation_chunk: RetrievedChunk
) -> None:
    llm, sdk = _llm()
    labels = {"provider": "openrouter", "model": MODEL}
    before = metric_value("llm.cost", **labels)

    await _service(llm, cancellation_chunk).ask(QUESTION)

    assert sdk.chat.completions.create.await_args.kwargs["extra_body"] == {"usage": {"include": True}}
    assert metric_value("llm.cost", **labels) - before == pytest.approx(0.0004)
    assert metric_value("gen_ai.client.token.usage", **{"gen_ai.token.type": "input"}) >= 1


async def test_another_endpoint_is_not_sent_openrouters_flag(cancellation_chunk: RetrievedChunk) -> None:
    llm, sdk = _llm("http://host.docker.internal:11434/v1", _completion(cost=None))

    await _service(llm, cancellation_chunk).ask(QUESTION)

    assert "extra_body" not in sdk.chat.completions.create.await_args.kwargs


async def test_a_provider_failure_is_recorded_with_the_cause_class_not_the_message(
    spans: InMemorySpanExporter, metric_value: MetricValue, cancellation_chunk: RetrievedChunk
) -> None:
    llm, sdk = _llm()
    sdk.chat.completions.create.side_effect = openai.APIConnectionError(
        request=httpx.Request("POST", OPENROUTER)
    )
    labels = {"gen_ai.request.model": MODEL, "error.type": "APIConnectionError"}
    before = metric_value("gen_ai.client.operation.duration", **labels)

    with pytest.raises(GenerationUnavailableError):
        await _service(llm, cancellation_chunk).ask(QUESTION)

    assert metric_value("gen_ai.client.operation.duration", **labels) - before == 1
    assert _attrs(spans, f"chat {MODEL}")["error.type"] == "APIConnectionError"


async def test_a_retry_is_counted_by_target_and_reason(metric_value: MetricValue) -> None:
    response = SimpleNamespace(headers={}, status_code=503)

    class _Unavailable(Exception):
        def __init__(self) -> None:
            super().__init__("503 with a message that must never become a label")
            self.response = response

    @retry_async(attempts=2, retry_on=_Unavailable, max_wait_seconds=0, target="test.target")
    async def call() -> None:
        raise _Unavailable

    before = metric_value("upstream.retries", target="test.target", reason="5xx")

    with pytest.raises(_Unavailable):
        await call()

    assert metric_value("upstream.retries", target="test.target", reason="5xx") - before == 1


async def test_each_request_logs_one_line_with_the_route_template_and_nothing_personal(
    cancellation_chunk: RetrievedChunk,
) -> None:
    llm, _ = _llm()
    service = _service(llm, cancellation_chunk)
    app = create_app()
    limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=60.0)
    app.dependency_overrides[get_rag_service] = lambda: service
    app.dependency_overrides[get_security_settings] = lambda: SecuritySettings(_env_file=None, api_key=API_KEY)
    app.dependency_overrides[get_rate_limiter] = lambda: limiter
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger = logging.getLogger("src.app.api.v1.middleware.request_id")
    logger.addHandler(handler)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("203.0.113.9", 1234)),
            base_url="http://test",
            headers={"X-API-Key": API_KEY},
        ) as client:
            await client.post("/api/v1/ask?token=hunter2", json={"question": QUESTION})
            await client.get("/no/such/thing/4f9c")
            await client.get("/health/live")
    finally:
        logger.removeHandler(handler)

    lines = [r for r in records if r.getMessage() == "http.request"]
    # The health probe is not logged: it would be one line every 30 seconds.
    assert [(r.method, r.route, r.status) for r in lines] == [  # type: ignore[attr-defined]
        ("POST", "/api/v1/ask", 200),
        ("GET", UNMATCHED_ROUTE, 404),
    ]
    flat = " ".join(str(value) for r in lines for value in r.__dict__.values())
    assert not any(secret in flat for secret in ("hunter2", "203.0.113.9", "Zorbak", "4f9c"))
