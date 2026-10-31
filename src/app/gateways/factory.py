import httpx
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

from src.app.core.request_id import get_request_id
from src.app.core.settings.ops_core import OpsCoreSettings
from src.app.core.settings.rerank import RerankSettings
from src.app.gateways.http_reranker import HttpReranker
from src.app.gateways.local_reranker import LocalReranker
from src.app.gateways.ops_core_knowledge import OpsCoreKnowledgeGateway

_API_KEY_HEADER = "X-API-Key"
_REQUEST_ID_HEADER = "X-Request-ID"


async def _inject_request_id(request: httpx.Request) -> None:
    """Carry this request's id upstream, so one id spans both services' logs.

    It cannot be a static header on the client: the client is built once for the
    process, the id changes per request. An event hook reads the ContextVar at
    send time, which is the only moment the right value exists.
    """
    if request_id := get_request_id():
        request.headers[_REQUEST_ID_HEADER] = request_id


def create_ops_core_knowledge_gateway(
    settings: OpsCoreSettings, transport: httpx.AsyncBaseTransport | None = None
) -> OpsCoreKnowledgeGateway:
    """Factory — the only place the ops-core-api HTTP client is constructed.

    Wires settings → a configured httpx.AsyncClient (base URL, auth header,
    timeout) → the gateway adapter, mirroring ``create_llm_client``. Keeping
    construction here lets the container stay declarative and the adapter stay
    purely I/O. `transport` lets a test run the real client, hooks and
    instrumentation included, against a stub instead of a server.
    """
    client = httpx.AsyncClient(
        base_url=settings.base_url.rstrip("/"),
        transport=transport,
        headers={_API_KEY_HEADER: settings.api_key.get_secret_value()},
        timeout=settings.timeout_seconds,
        event_hooks={"request": [_inject_request_id]},
    )
    # A client span per attempt and a `traceparent`, so ops-core-api's server span is a
    # child of this request's trace. The X-Request-ID hook stays: logs join on it.
    HTTPXClientInstrumentor().instrument_client(client)
    return OpsCoreKnowledgeGateway(client=client, max_attempts=settings.max_attempts)


def create_reranker(settings: RerankSettings) -> HttpReranker | LocalReranker | None:
    """None when reranking is switched off, so the service skips the step entirely."""
    if not settings.enabled:
        return None
    if settings.provider == "local":
        return LocalReranker(model_name=settings.local_model)
    if settings.api_key is None:
        raise AssertionError("unreachable: RerankSettings rejects an enabled http reranker without a key")
    client = httpx.AsyncClient(
        base_url=settings.base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {settings.api_key.get_secret_value()}"},
        timeout=settings.timeout_seconds,
    )
    # Third-party API: the `traceparent` it receives carries only random ids, and the
    # client span is how a slow rerank is seen.
    HTTPXClientInstrumentor().instrument_client(client)
    return HttpReranker(client=client, model=settings.model)
