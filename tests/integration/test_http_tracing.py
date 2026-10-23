"""The HTTP server spans and metrics under FastAPI 0.139 (CANON C19).

`instrument_app` replaces a private function of opentelemetry-instrumentation-fastapi 0.60b1,
which cannot read this FastAPI's nested routers. These cases fail if that patch is dropped
too early or stops working: a wrong method must be a 405 rather than a 500, and the route
on the span and the metric must be the template rather than the raw path.
"""

from collections.abc import Callable

import httpx
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.main import create_app


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test")


def _server_spans(spans: InMemorySpanExporter) -> list[dict[str, object]]:
    return [
        {"name": span.name, **(span.attributes or {})}
        for span in spans.get_finished_spans()
        if span.kind == trace.SpanKind.SERVER
    ]


async def test_a_wrong_method_is_a_405_and_the_route_is_the_template(
    spans: InMemorySpanExporter, metric_value: Callable[..., float]
) -> None:
    labels = {"http.route": "/api/v1/ask", "http.response.status_code": 405}
    before = metric_value("http.server.request.duration", **labels)

    async with _client() as client:
        response = await client.put("/api/v1/ask")

    assert response.status_code == 405
    (span,) = _server_spans(spans)
    assert span["name"] == "PUT /api/v1/ask"
    assert span["http.route"] == "/api/v1/ask"
    # The stable HTTP names, which the dashboards are written against.
    assert metric_value("http.server.request.duration", **labels) - before == 1


async def test_a_cors_preflight_is_not_a_500(spans: InMemorySpanExporter) -> None:
    async with _client() as client:
        response = await client.options(
            "/api/v1/ask",
            headers={"Origin": "http://example.test", "Access-Control-Request-Method": "POST"},
        )

    assert response.status_code < 500


async def test_an_unknown_path_with_an_id_in_it_does_not_become_a_label(
    spans: InMemorySpanExporter, metric_value: Callable[..., float]
) -> None:
    path = "/api/v1/documents/0af7651916cd43dd8448eb211c80319c"

    async with _client() as client:
        response = await client.get(path)

    assert response.status_code == 404
    (span,) = _server_spans(spans)
    assert "http.route" not in span
    assert path not in str(span["name"])
    assert metric_value("http.server.request.duration", **{"http.route": path}) == 0
