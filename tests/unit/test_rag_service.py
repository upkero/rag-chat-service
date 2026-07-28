from unittest.mock import AsyncMock

import pytest

from src.app.contracts.llm.llm_response import LLMResponse
from src.app.contracts.rag.retrieved_chunk import RetrievedChunk
from src.app.exceptions.llm import GenerationUnavailableError, LLMGenerationError
from src.app.messages import get_message
from src.app.services.rag import RagService


async def test_ask_confident_calls_llm_and_returns_sources(
    rag_service: RagService,
    gateway: AsyncMock,
    llm_client: AsyncMock,
    cancellation_chunk: RetrievedChunk,
) -> None:
    """DoD scenario 1: 'how do I cancel' -> precise answer + its source."""
    gateway.search.return_value = [cancellation_chunk]
    llm_client.complete.return_value = LLMResponse(
        content="You can cancel free of charge up to twenty-four hours before the start time.",
        usage={"total_tokens": 42},
    )

    answer = await rag_service.ask("how do I cancel an appointment")

    gateway.search.assert_awaited_once_with("how do I cancel an appointment", 5)
    llm_client.complete.assert_awaited_once()
    assert "twenty-four hours" in answer.answer
    assert len(answer.sources) == 1
    source = answer.sources[0]
    assert (source.document_title, source.chunk_index, source.score) == (
        "Cancellation and rescheduling policy",
        0,
        0.82,
    )


async def test_ask_below_threshold_skips_llm_and_returns_fallback(
    rag_service: RagService,
    gateway: AsyncMock,
    llm_client: AsyncMock,
) -> None:
    """DoD scenario 2: off-topic question is rejected WITHOUT calling the LLM."""
    gateway.search.return_value = [
        RetrievedChunk(document_title="Treatments and therapies", chunk_index=0, chunk_text="...", score=0.2),
    ]

    answer = await rag_service.ask("what is the weather on mars")

    llm_client.complete.assert_not_awaited()
    assert answer.answer == get_message("low_confidence", "en")
    assert answer.sources == ()


async def test_ask_no_chunks_skips_llm(
    rag_service: RagService,
    gateway: AsyncMock,
    llm_client: AsyncMock,
) -> None:
    gateway.search.return_value = []

    answer = await rag_service.ask("something nobody ever wrote a document about")

    llm_client.complete.assert_not_awaited()
    assert answer.sources == ()


async def test_ask_wraps_llm_failure_as_generation_unavailable(
    rag_service: RagService,
    gateway: AsyncMock,
    llm_client: AsyncMock,
    cancellation_chunk: RetrievedChunk,
) -> None:
    """Generate-phase provider failure -> typed 502, not a raw error."""
    gateway.search.return_value = [cancellation_chunk]
    llm_client.complete.side_effect = LLMGenerationError("provider down")

    with pytest.raises(GenerationUnavailableError):
        await rag_service.ask("how do I cancel an appointment")


async def test_generate_prompt_carries_language_directive_and_context(
    rag_service: RagService,
    gateway: AsyncMock,
    llm_client: AsyncMock,
    cancellation_chunk: RetrievedChunk,
) -> None:
    """Russian question over the English knowledge base — the realistic direction.

    The LLM is mocked, so we can't assert it actually translates — instead we
    assert the wiring: the rendered prompt carries the 'answer in the question's
    language' rule and the English context reaches the model intact.
    """
    gateway.search.return_value = [cancellation_chunk]
    llm_client.complete.return_value = LLMResponse(content="Отменить можно бесплатно за сутки.")

    answer = await rag_service.ask("как отменить запись")

    assert answer.answer == "Отменить можно бесплатно за сутки."
    messages = llm_client.complete.await_args.args[0]
    sent = "\n".join(message.content for message in messages)
    # The language rule is a rendered placeholder, not a translated prompt file.
    assert "the same language the question is written in" in sent
    assert "twenty-four hours" in sent  # context passed through
    assert "как отменить запись" in sent
