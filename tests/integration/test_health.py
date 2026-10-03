from unittest.mock import AsyncMock

import httpx
import pytest

from src.app.api.v1.dependencies import get_knowledge_gateway, get_llm_client
from src.app.bootstrap.container import ApplicationContainer
from src.app.interfaces.knowledge.knowledge_gateway import KnowledgeGateway
from src.app.interfaces.llm.llm_client import LLMClient
from src.main import create_app


def _client(*, llm_ok: bool, ops_core_ok: bool) -> httpx.AsyncClient:
    app = create_app()
    llm = AsyncMock(spec=LLMClient)
    llm.ping.return_value = llm_ok
    gateway = AsyncMock(spec=KnowledgeGateway)
    gateway.ping.return_value = ops_core_ok
    app.dependency_overrides[get_llm_client] = lambda: llm
    app.dependency_overrides[get_knowledge_gateway] = lambda: gateway
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_liveness_is_open_and_ok() -> None:
    # /health is outside /api/v1, so it needs no API key.
    async with _client(llm_ok=True, ops_core_ok=True) as client:
        response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_readiness_ok_when_all_deps_up() -> None:
    async with _client(llm_ok=True, ops_core_ok=True) as client:
        response = await client.get("/health/ready")
    assert response.status_code == 200


async def test_readiness_503_when_ops_core_down() -> None:
    async with _client(llm_ok=True, ops_core_ok=False) as client:
        response = await client.get("/health/ready")
    assert response.status_code == 503
    assert "ops_core" in response.json()["detail"]


async def test_a_misconfiguration_fails_the_boot_not_the_first_request(monkeypatch: pytest.MonkeyPatch) -> None:
    # A container property that cannot be built stands in for any misconfiguration.
    def broken(self: ApplicationContainer) -> None:
        raise RuntimeError("misconfigured")

    monkeypatch.setattr(ApplicationContainer, "rag_service", property(broken))
    app = create_app()

    with pytest.raises(RuntimeError, match="misconfigured"):
        async with app.router.lifespan_context(app):
            pass
