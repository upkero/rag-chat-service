import httpx

from src.app.core.settings.ops_core import OpsCoreSettings
from src.app.gateways.ops_core_knowledge import OpsCoreKnowledgeGateway

_API_KEY_HEADER = "X-API-Key"


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
    )
    return OpsCoreKnowledgeGateway(client=client, max_attempts=settings.max_attempts)
