"""The spans an incident is read from, and the log fields that lead to them.

Every case drives the real code: the app factory, the real ops-core gateway built by
the production factory, and the real LLM client, against stubbed transports.
"""

import json
import logging
from collections.abc import Iterator
from unittest.mock import AsyncMock

import httpx
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.app.api.v1.dependencies import get_rag_service
from src.app.api.v1.dependencies.rate_limit import get_rate_limiter
from src.app.core.rate_limit import SlidingWindowRateLimiter
from src.app.core.settings.ops_core import OpsCoreSettings
from src.app.core.settings.rag import RagSettings
from src.app.core.settings.security import SecuritySettings, get_security_settings
from src.app.core.telemetry import setup_tracing
from src.app.gateways.factory import create_ops_core_knowledge_gateway
from src.app.interfaces.llm.llm_client import LLMClient
from src.app.services.rag import RagService
from src.main import create_app

API_KEY = "test-api-key-0123456789"
TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
TRACEPARENT = f"00-{TRACE_ID}-b7ad6b7169203331-01"

# Set once per process: OTel refuses to replace a global provider.
_exporter = InMemorySpanExporter()


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    provider = setup_tracing("rag-chat-service", "rag-chat-service")
    assert isinstance(provider, TracerProvider)
    if not getattr(provider, "_test_exporter_added", False):
        provider.add_span_processor(SimpleSpanProcessor(_exporter))
        provider._test_exporter_added = True  # type: ignore[attr-defined]
    _exporter.clear()
    yield _exporter
    _exporter.clear()


def _search_stub(seen: list[httpx.Request]) -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"matches": []})

    return httpx.MockTransport(respond)


async def test_a_request_is_one_trace_from_the_caller_through_to_ops_core(spans: InMemorySpanExporter) -> None:
    seen: list[httpx.Request] = []
    gateway = create_ops_core_knowledge_gateway(
        OpsCoreSettings(_env_file=None, api_key=API_KEY, base_url="http://core.test"), _search_stub(seen)
    )
    service = RagService(
        knowledge=gateway, llm_client=AsyncMock(spec=LLMClient), settings=RagSettings(_env_file=None)
    )
    app = create_app()
    limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=60.0)
    app.dependency_overrides[get_rag_service] = lambda: service
    app.dependency_overrides[get_security_settings] = lambda: SecuritySettings(_env_file=None, api_key=API_KEY)
    app.dependency_overrides[get_rate_limiter] = lambda: limiter

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test", headers={"X-API-Key": API_KEY}
    ) as client:
        response = await client.post(
            "/api/v1/ask",
            json={"question": "how do I cancel"},
            headers={"traceparent": TRACEPARENT, "X-Request-ID": "req-123"},
        )
    await gateway.close()

    assert response.status_code == 200
    server = next(span for span in spans.get_finished_spans() if span.kind == trace.SpanKind.SERVER)
    assert format(server.context.trace_id, "032x") == TRACE_ID
    assert server.attributes is not None
    assert server.attributes["request_id"] == "req-123"
    # The outgoing search carries a traceparent of the same trace, so ops-core's server
    # span becomes a child of this request.
    assert seen[0].headers["traceparent"].split("-")[1] == TRACE_ID
    assert seen[0].headers["X-Request-ID"] == "req-123"


def test_log_lines_carry_the_trace_and_span_ids_inside_a_span(spans: InMemorySpanExporter) -> None:
    from src.app.core.logging import _JsonFormatter, _RequestIdFilter

    def render() -> dict[str, str]:
        record = logging.LogRecord("t", logging.INFO, "", 0, "msg", (), None)
        _RequestIdFilter().filter(record)
        return json.loads(_JsonFormatter().format(record))  # type: ignore[no-any-return]

    outside = render()
    with trace.get_tracer(__name__).start_as_current_span("x") as span:
        inside = render()

    assert outside["trace_id"] == "" and outside["span_id"] == ""
    assert inside["trace_id"] == format(span.get_span_context().trace_id, "032x")
    assert inside["span_id"] == format(span.get_span_context().span_id, "016x")
