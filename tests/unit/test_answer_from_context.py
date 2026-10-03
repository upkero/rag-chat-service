from unittest.mock import AsyncMock

import pytest

from src.app.contracts.llm.llm_response import LLMResponse
from src.app.contracts.rag.retrieved_chunk import RetrievedChunk
from src.app.interfaces.llm.llm_client import LLMClient
from src.app.llm.skills.answer_from_context import NO_ANSWER, answer_from_context
from src.app.prompts import get_prompt


async def _ask(content: str, chunk: RetrievedChunk):
    llm = AsyncMock(spec=LLMClient)
    llm.complete.return_value = LLMResponse(content=content, usage={})
    return await answer_from_context(client=llm, question="q", chunks=[chunk])


@pytest.mark.parametrize(
    "reply",
    [
        "NO_ANSWER",
        "  NO_ANSWER\n",
        "**NO_ANSWER**",
        "NO_ANSWER.",
        '"NO_ANSWER"',
        "`NO_ANSWER`",
        "no_answer",
        "\n\nNO_ANSWER\n\n",
        "_NO_ANSWER_",
    ],
)
async def test_a_reply_that_is_only_the_marker_is_a_refusal(reply: str, cancellation_chunk: RetrievedChunk) -> None:
    assert await _ask(reply, cancellation_chunk) is None


@pytest.mark.parametrize(
    "reply",
    [
        "On Saturday we are open from nine until six. NO_ANSWER",
        "On Saturday we are open from nine until six.\n\nNO_ANSWER",
        "On Saturday we are open from nine until six. **NO_ANSWER**",
        "On Saturday we are open from nine until six. NO_ANSWER.",
        "NO_ANSWER\nOn Saturday we are open from nine until six.",
    ],
)
async def test_the_marker_is_removed_from_a_partial_answer(reply: str, cancellation_chunk: RetrievedChunk) -> None:
    result = await _ask(reply, cancellation_chunk)

    assert result is not None
    assert result.answer == "On Saturday we are open from nine until six."


async def test_a_reply_without_the_marker_is_left_alone(cancellation_chunk: RetrievedChunk) -> None:
    result = await _ask("Up to 24 hours before.", cancellation_chunk)

    assert result is not None
    assert result.answer == "Up to 24 hours before."


async def test_the_prompt_gets_its_marker_from_the_code_constant(cancellation_chunk: RetrievedChunk) -> None:
    llm = AsyncMock(spec=LLMClient)
    llm.complete.return_value = LLMResponse(content="fine", usage={})

    await answer_from_context(client=llm, question="q", chunks=[cancellation_chunk])

    sent = llm.complete.await_args.args[0][0].content
    assert sent.count(NO_ANSWER) == 2  # the instruction and the "never append it" rule
    assert "never add" in get_prompt("rag.answer").text
