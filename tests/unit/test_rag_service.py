from unittest.mock import AsyncMock

import pytest

from src.app.contracts.llm.llm_response import LLMResponse
from src.app.contracts.rag.retrieved_chunk import RetrievedChunk
from src.app.exceptions.llm import GenerationUnavailableError, LLMGenerationError
from src.app.services.rag_service import RagService


async def test_ask_confident_calls_llm_and_returns_sources(
    rag_service: RagService,
    retriever: AsyncMock,
    llm_client: AsyncMock,
    mesotherapy_chunk: RetrievedChunk,
) -> None:
    """DoD scenario 1: 'сколько стоит мезотерапия' -> precise answer + source."""
    retriever.search.return_value = [mesotherapy_chunk]
    llm_client.complete.return_value = LLMResponse(
        content="Мезотерапия лица стоит 3500 рублей.",
        usage={"total_tokens": 42},
    )

    answer = await rag_service.ask("сколько стоит мезотерапия")

    retriever.search.assert_awaited_once_with("сколько стоит мезотерапия", 5)
    llm_client.complete.assert_awaited_once()
    assert answer.from_llm is True
    assert "3500" in answer.answer
    assert len(answer.sources) == 1
    source = answer.sources[0]
    assert (source.document_title, source.chunk_index, source.score) == ("Прайс-лист", 2, 0.82)


async def test_ask_below_threshold_skips_llm_and_returns_fallback(
    rag_service: RagService,
    retriever: AsyncMock,
    llm_client: AsyncMock,
) -> None:
    """DoD scenario 2: off-topic question is rejected WITHOUT calling the LLM."""
    retriever.search.return_value = [
        RetrievedChunk(document_title="FAQ", chunk_index=0, chunk_text="...", score=0.2),
    ]

    answer = await rag_service.ask("какая погода на марсе")

    llm_client.complete.assert_not_awaited()
    assert answer.from_llm is False
    assert answer.answer == "Не могу ответить точно, уточните у администратора."
    assert answer.sources == ()


async def test_ask_no_chunks_skips_llm(
    rag_service: RagService,
    retriever: AsyncMock,
    llm_client: AsyncMock,
) -> None:
    retriever.search.return_value = []

    answer = await rag_service.ask("что-то совсем не по теме")

    llm_client.complete.assert_not_awaited()
    assert answer.from_llm is False
    assert answer.sources == ()


async def test_ask_wraps_llm_failure_as_generation_unavailable(
    rag_service: RagService,
    retriever: AsyncMock,
    llm_client: AsyncMock,
    mesotherapy_chunk: RetrievedChunk,
) -> None:
    """Generate-phase provider failure -> typed 502, not a raw error."""
    retriever.search.return_value = [mesotherapy_chunk]
    llm_client.complete.side_effect = LLMGenerationError("provider down")

    with pytest.raises(GenerationUnavailableError):
        await rag_service.ask("сколько стоит мезотерапия")


async def test_generate_prompt_carries_language_directive_and_context(
    rag_service: RagService,
    retriever: AsyncMock,
    llm_client: AsyncMock,
) -> None:
    """English question over a Russian source chunk.

    The LLM is mocked, so we can't assert it actually translates — instead we
    assert the wiring: the system prompt carries the 'answer in the question's
    language' rule and the Russian context reaches the model.
    """
    retriever.search.return_value = [
        RetrievedChunk(
            document_title="Price list",
            chunk_index=0,
            chunk_text="Мезотерапия лица — 3500 рублей.",
            score=0.9,
        ),
    ]
    llm_client.complete.return_value = LLMResponse(content="Mesotherapy costs 3500 RUB.")

    answer = await rag_service.ask("how much is mesotherapy?")

    assert answer.answer == "Mesotherapy costs 3500 RUB."
    messages = llm_client.complete.await_args.args[0]
    system_message = next(message for message in messages if message.role == "system")
    user_message = next(message for message in messages if message.role == "user")
    assert "на том же языке" in system_message.content  # language-fidelity rule present
    assert "3500" in user_message.content  # Russian context passed through
