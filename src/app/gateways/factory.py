import httpx

from src.app.core.request_id import get_request_id
from src.app.core.settings.ops_core import OpsCoreSettings
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


def create_ops_core_knowledge_gateway(settings: OpsCoreSettings) -> OpsCoreKnowledgeGateway:
    """Factory — the only place the ops-core-api HTTP client is constructed.

    Wires settings → a configured httpx.AsyncClient (base URL, auth header,
    timeout) → the gateway adapter, mirroring ``create_llm_client``. Keeping
    construction here lets the container stay declarative and the adapter stay
    purely I/O.
    """
    client = httpx.AsyncClient(
        base_url=settings.base_url.rstrip("/"),
        headers={_API_KEY_HEADER: settings.api_key.get_secret_value()},
        timeout=settings.timeout_seconds,
        event_hooks={"request": [_inject_request_id]},
    )
    return OpsCoreKnowledgeGateway(client=client, max_attempts=settings.max_attempts)
