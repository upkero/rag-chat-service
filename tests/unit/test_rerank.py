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


async def test_a_no_answer_from_the_model_becomes_a_refusal_without_sources(
    gateway: AsyncMock, llm_client: AsyncMock, cancellation_chunk: RetrievedChunk
) -> None:
    # Small talk can pass the score gate ("как дела" scores ~0.09); the model then says NO_ANSWER.
    gateway.search.return_value = [replace(cancellation_chunk, score=0.9)]
    llm_client.complete.return_value = LLMResponse(content="NO_ANSWER", usage={})

    answer = await RagService(
        knowledge=gateway, llm_client=llm_client, settings=RagSettings(message_language="en")
    ).ask("как дела")

    assert answer.answer == "Не могу ответить точно, уточните у администратора."
    assert answer.sources == ()


async def _ask_with_scores(
    gateway: AsyncMock, llm: AsyncMock, chunk: RetrievedChunk, scores: list[float], *, rerank: bool
):
    chunks = [replace(chunk, chunk_index=index, score=score) for index, score in enumerate(scores)]
    gateway.search.return_value = chunks
    llm.complete.return_value = LLMResponse(content="Free up to 24 hours before.", usage={})
    if not rerank:
        return await RagService(
            knowledge=gateway, llm_client=llm, settings=RagSettings(score_threshold=0.5, max_context_chunks=5)
        ).ask("how do I cancel")
    reranker = AsyncMock(spec=Reranker)
    reranker.rerank.return_value = chunks
    return await RagService(
        knowledge=gateway,
        llm_client=llm,
        settings=RagSettings(score_threshold=0.5, max_context_chunks=5),
        reranker=reranker,
        rerank_settings=RerankSettings(_env_file=None, min_score=0.08),
    ).ask("how do I cancel")


async def test_irrelevant_chunks_are_not_sources_or_context(
    gateway: AsyncMock, llm_client: AsyncMock, cancellation_chunk: RetrievedChunk
) -> None:
    # The live demo: best 0.674, then a tail the old code cited and fed to the model.
    answer = await _ask_with_scores(
        gateway, llm_client, cancellation_chunk, [0.674, 0.204, 0.165, 0.0804, 0.0743], rerank=True
    )

    assert [source.score for source in answer.sources] == [0.674]
    prompt = "\n".join(message.content for message in llm_client.complete.await_args.args[0])
    assert prompt.count("[1]") == 1
    assert "[2]" not in prompt


async def test_chunks_near_the_best_one_are_kept(
    gateway: AsyncMock, llm_client: AsyncMock, cancellation_chunk: RetrievedChunk
) -> None:
    answer = await _ask_with_scores(gateway, llm_client, cancellation_chunk, [0.6, 0.4, 0.25, 0.1], rerank=True)

    assert [source.score for source in answer.sources] == [0.6, 0.4]  # 0.25 is under half of 0.6


async def test_the_embedding_gate_filters_sources_without_a_reranker(
    gateway: AsyncMock, llm_client: AsyncMock, cancellation_chunk: RetrievedChunk
) -> None:
    answer = await _ask_with_scores(gateway, llm_client, cancellation_chunk, [0.82, 0.7, 0.45], rerank=False)

    assert [source.score for source in answer.sources] == [0.82, 0.7]  # 0.45 < RAG_SCORE_THRESHOLD
