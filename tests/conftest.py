from unittest.mock import AsyncMock

import pytest

from src.app.contracts.rag.retrieved_chunk import RetrievedChunk
from src.app.core.settings.rag import RagSettings
from src.app.interfaces.knowledge.chunk_retriever import ChunkRetriever
from src.app.interfaces.llm.llm_client import LLMClient
from src.app.services.rag_service import RagService


@pytest.fixture
def rag_settings() -> RagSettings:
    # Explicit values so tests never depend on a local .env.
    return RagSettings(
        score_threshold=0.5,
        top_k=5,
        max_context_chunks=5,
        fallback_message="Не могу ответить точно, уточните у администратора.",
    )


@pytest.fixture
def retriever() -> AsyncMock:
    # spec=ChunkRetriever => the mock conforms to the interface (Liskov-safe double).
    return AsyncMock(spec=ChunkRetriever)


@pytest.fixture
def llm_client() -> AsyncMock:
    return AsyncMock(spec=LLMClient)


@pytest.fixture
def rag_service(
    retriever: AsyncMock,
    llm_client: AsyncMock,
    rag_settings: RagSettings,
) -> RagService:
    return RagService(retriever=retriever, llm_client=llm_client, settings=rag_settings)


@pytest.fixture
def mesotherapy_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        document_title="Прайс-лист",
        chunk_index=2,
        chunk_text="Мезотерапия лица — 3500 рублей за процедуру.",
        score=0.82,
    )
