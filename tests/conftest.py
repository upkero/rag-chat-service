from collections.abc import Callable, Iterator
from unittest.mock import AsyncMock

import pytest
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.app.contracts.rag.retrieved_chunk import RetrievedChunk
from src.app.core.settings.app import AppSettings
from src.app.core.settings.logging import LoggingSettings
from src.app.core.settings.rag import RagSettings
from src.app.core.telemetry import setup_tracing
from src.app.interfaces.knowledge.knowledge_gateway import KnowledgeGateway
from src.app.interfaces.llm.llm_client import LLMClient
from src.app.services.rag import RagService

# Installed at import, before any test module imports `src.main` (which builds the app and
# would install a provider with no reader): OTel refuses to replace a global provider.
_metric_reader = InMemoryMetricReader()
metrics.set_meter_provider(MeterProvider(metric_readers=[_metric_reader]))
_span_exporter = InMemorySpanExporter()


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    provider = setup_tracing("rag-chat-service", "rag-chat-service")
    assert isinstance(provider, TracerProvider)
    if not getattr(provider, "_test_exporter_added", False):
        provider.add_span_processor(SimpleSpanProcessor(_span_exporter))
        provider._test_exporter_added = True  # type: ignore[attr-defined]
    _span_exporter.clear()
    yield _span_exporter
    _span_exporter.clear()


@pytest.fixture
def metric_value() -> Callable[..., float]:
    """The running total of a counter, or the sample count of a histogram, for the data
    points whose attributes include `attrs`. Totals are cumulative across the whole test
    run, so a test compares a value before and after, never against zero."""

    def read(name: str, **attrs: object) -> float:
        total = 0.0
        data = _metric_reader.get_metrics_data()
        for resource_metrics in data.resource_metrics if data else ():
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    if metric.name != name:
                        continue
                    for point in metric.data.data_points:
                        point_attrs = point.attributes or {}
                        if all(point_attrs.get(key) == value for key, value in attrs.items()):
                            total += getattr(point, "value", None) or getattr(point, "count", 0) or 0
        return total

    return read


@pytest.fixture(autouse=True)
def _hermetic_app_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    # create_app() reads these two from `.env` too; a developer's file must not leak in.
    monkeypatch.setattr("src.main.get_app_settings", lambda: AppSettings(_env_file=None))
    monkeypatch.setattr("src.main.get_logging_settings", lambda: LoggingSettings(_env_file=None))


@pytest.fixture
def rag_settings() -> RagSettings:
    # Explicit values so tests never depend on a local .env.
    return RagSettings(
        _env_file=None,
        score_threshold=0.5,
        top_k=5,
        max_context_chunks=5,
    )


@pytest.fixture
def gateway() -> AsyncMock:
    # spec=KnowledgeGateway => the mock conforms to the interface (Liskov-safe double).
    return AsyncMock(spec=KnowledgeGateway)


@pytest.fixture
def llm_client() -> AsyncMock:
    return AsyncMock(spec=LLMClient)


@pytest.fixture
def rag_service(
    gateway: AsyncMock,
    llm_client: AsyncMock,
    rag_settings: RagSettings,
) -> RagService:
    return RagService(knowledge=gateway, llm_client=llm_client, settings=rag_settings)


@pytest.fixture
def cancellation_chunk() -> RetrievedChunk:
    # Copied from ops-core-api's seed data (cli/seed_data.py), so every example in
    # the tests and the README reproduces against a seeded backend. A fixture
    # describing content no seed contains only ever returns the fallback message.
    return RetrievedChunk(
        document_title="Cancellation and rescheduling policy",
        chunk_index=0,
        chunk_text=(
            "Appointments can be cancelled or rescheduled free of charge up to "
            "twenty-four hours before the scheduled start time."
        ),
        score=0.82,
    )
