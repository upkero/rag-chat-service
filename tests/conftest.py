from unittest.mock import AsyncMock

import pytest

from src.app.contracts.rag.retrieved_chunk import RetrievedChunk
from src.app.core.settings.app import AppSettings
from src.app.core.settings.logging import LoggingSettings
from src.app.core.settings.rag import RagSettings
from src.app.interfaces.knowledge.knowledge_gateway import KnowledgeGateway
from src.app.interfaces.llm.llm_client import LLMClient
from src.app.services.rag import RagService


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
