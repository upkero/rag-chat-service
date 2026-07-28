"""Skill: answer a question strictly from retrieved knowledge-base context.

The one unit of model work this service performs. It owns the prompt, calls the
model through the `LLMClient` port and hands back a `RagAnswer` — never the raw
string, so no caller ever has to guess what the model meant.
"""

from collections.abc import Sequence
from logging import getLogger

from src.app.contracts.rag.rag_answer import RagAnswer
from src.app.contracts.rag.retrieved_chunk import RetrievedChunk
from src.app.exceptions.llm import GenerationUnavailableError, LLMError
from src.app.interfaces.llm.llm_client import LLMClient
from src.app.llm.prompt_builder import PromptBuilder
from src.app.prompts import get_prompt

logger = getLogger(__name__)

_PROMPT = get_prompt("rag.answer")
# The question decides the language, which is what the Russian original said in
# Russian. Stated as a value rather than a translated file: one prompt, and the
# day a deployment has to answer only in German it is this argument that changes.
_QUESTION_LANGUAGE = "the same language the question is written in"


async def answer_from_context(
    *,
    client: LLMClient,
    question: str,
    chunks: Sequence[RetrievedChunk],
    reply_language: str = _QUESTION_LANGUAGE,
) -> RagAnswer:
    """Ground the model on ``chunks`` and answer ``question`` from them alone."""
    rendered = _PROMPT.render(
        reply_language=reply_language,
        context=_format_context(chunks),
        question=question,
    )

    try:
        response = await client.complete(PromptBuilder().system(rendered).build())
    except LLMError as exc:
        # Symmetric to the retrieval phase: any provider failure (rate limit,
        # timeout, network) becomes a typed 502 rather than a raw error reaching
        # the router.
        raise GenerationUnavailableError("LLM provider failed to generate an answer.") from exc

    usage = response.usage or {}
    # The prompt id travels with the answer: when a reply looks wrong six weeks
    # from now, this is what says whether the wording had already been edited.
    logger.info(
        "rag.generate",
        extra={
            "prompt_id": _PROMPT.id,
            "context_chunks": len(chunks),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "model": response.model,
        },
    )
    return RagAnswer(answer=response.content, sources=tuple(chunks), from_llm=True)


def _format_context(chunks: Sequence[RetrievedChunk]) -> str:
    # Numbered so the model can refer to an excerpt, titled so it knows which
    # document a claim came from.
    return "\n\n".join(
        f"[{index}] {chunk.document_title}\n{chunk.chunk_text}"
        for index, chunk in enumerate(chunks, start=1)
    )
