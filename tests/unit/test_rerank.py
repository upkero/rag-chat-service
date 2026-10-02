from dataclasses import replace
from unittest.mock import AsyncMock

from src.app.contracts.llm.llm_response import LLMResponse
from src.app.contracts.rag.retrieved_chunk import RetrievedChunk
from src.app.core.settings.rag import RagSettings
from src.app.core.settings.rerank import RerankSettings
from src.app.interfaces.rerank.reranker import Reranker
from src.app.services.rag import RagService


def _service(gateway: AsyncMock, llm: AsyncMock, reranker: AsyncMock) -> RagService:
    return RagService(
        knowledge=gateway,
        llm_client=llm,
        settings=RagSettings(score_threshold=0.5, top_k=5, max_context_chunks=3),
        reranker=reranker,
        rerank_settings=RerankSettings(candidates=10, min_score=0.1),
    )


async def test_rerank_score_replaces_the_cosine_gate(
    gateway: AsyncMock, llm_client: AsyncMock, cancellation_chunk: RetrievedChunk
) -> None:
    # Cosine 0.2 would be refused by the 0.5 gate; the cross-encoder says 0.9.
    weak = replace(cancellation_chunk, score=0.2)
    reranker = AsyncMock(spec=Reranker)
    reranker.rerank.return_value = [replace(weak, score=0.9)]
    gateway.search.return_value = [weak]
    llm_client.complete.return_value = LLMResponse(content="Free up to 24 hours before.", usage={})

    answer = await _service(gateway, llm_client, reranker).ask("how do I cancel")

    gateway.search.assert_awaited_once_with("how do I cancel", 10)
    llm_client.complete.assert_awaited_once()
    assert answer.sources[0].score == 0.9


async def test_a_failing_reranker_falls_back_to_the_embedding_gate(
    gateway: AsyncMock, llm_client: AsyncMock, cancellation_chunk: RetrievedChunk
) -> None:
    reranker = AsyncMock(spec=Reranker)
    reranker.rerank.side_effect = RuntimeError("down")
    gateway.search.return_value = [replace(cancellation_chunk, score=0.2)]

    answer = await _service(gateway, llm_client, reranker).ask("how do I cancel")

    llm_client.complete.assert_not_awaited()  # 0.2 < 0.5, as it was before reranking existed
    assert answer.sources == ()


def test_local_provider_needs_no_api_key() -> None:
    settings = RerankSettings(_env_file=None, enabled=True, provider="local")

    assert settings.local_model == "BAAI/bge-reranker-v2-m3"


def test_http_provider_still_requires_a_key() -> None:
    import pytest

    with pytest.raises(ValueError, match="RERANK_API_KEY"):
        RerankSettings(_env_file=None, enabled=True, provider="http")


async def test_a_russian_question_gets_a_russian_refusal(gateway: AsyncMock, llm_client: AsyncMock) -> None:
    gateway.search.return_value = []

    answer = await RagService(
        knowledge=gateway, llm_client=llm_client, settings=RagSettings(message_language="en")
    ).ask("Как приготовить борщ?")

    assert answer.answer == "Не могу ответить точно, уточните у администратора."
