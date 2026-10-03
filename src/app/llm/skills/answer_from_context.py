"""Skill: answer a question strictly from retrieved knowledge-base context.

The one unit of model work this service performs. It owns the prompt, calls the
model through the `LLMClient` port and hands back a `RagAnswer` — never the raw
string, so no caller ever has to guess what the model meant.
"""

import re
from collections.abc import Sequence
from logging import getLogger

from src.app.contracts.rag.rag_answer import RagAnswer
from src.app.contracts.rag.retrieved_chunk import RetrievedChunk
from src.app.exceptions.llm import GenerationUnavailableError, LLMError
from src.app.interfaces.llm.llm_client import LLMClient
from src.app.llm.prompt_builder import PromptBuilder
from src.app.prompts import get_prompt

logger = getLogger(__name__)

# Rules are the system message; the retrieved excerpts and the question are user-role data.
# Document text can be uploaded by anyone holding an ops-core key and the question comes from
# the public, so neither gets system authority.
_SYSTEM_PROMPT = get_prompt("rag.answer.system")
_USER_PROMPT = get_prompt("rag.answer.user")
# The question decides the language, which is what the Russian original said in
# Russian. Stated as a value rather than a translated file: one prompt, and the
# day a deployment has to answer only in German it is this argument that changes.
_QUESTION_LANGUAGE = "the same language the question is written in"
# What the prompt tells the model to say when the context holds no answer. A marker rather
# than prose so the caller can swap in the localised canned refusal and drop the sources.
NO_ANSWER = "NO_ANSWER"
# Models rarely emit the marker bare: they bold it, quote it, end it with a full stop or
# bolt it onto a partial answer. Any such form is stripped wherever it stands.
_MARKER = re.compile(rf"[*_`\"']*(?<![A-Za-z]){NO_ANSWER}(?![A-Za-z])[*_`\"']*\.?", re.IGNORECASE)


async def answer_from_context(
    *,
    client: LLMClient,
    question: str,
    chunks: Sequence[RetrievedChunk],
    reply_language: str = _QUESTION_LANGUAGE,
) -> RagAnswer | None:
    """Ground the model on ``chunks`` and answer ``question`` from them alone.

    None means the model found no answer in the context."""
    messages = (
        PromptBuilder()
        .system(_SYSTEM_PROMPT.render(reply_language=reply_language, no_answer_marker=NO_ANSWER))
        .user(_USER_PROMPT.render(context=_format_context(chunks), question=question))
        .build()
    )

    try:
        response = await client.complete(messages)
    except LLMError as exc:
        # Symmetric to the retrieval phase: any provider failure (rate limit,
        # timeout, network) becomes a typed 502 rather than a raw error reaching
        # the router.
        raise GenerationUnavailableError("LLM provider failed to generate an answer.") from exc

    usage = response.usage or {}
    # The prompt ids travel with the answer: when a reply looks wrong six weeks
    # from now, this is what says whether the wording had already been edited.
    logger.info(
        "rag.generate",
        extra={
            "prompt_id": _SYSTEM_PROMPT.id,
            "user_prompt_id": _USER_PROMPT.id,
            "context_chunks": len(chunks),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "model": response.model,
        },
    )
    answer = _MARKER.sub("", response.content).strip()
    # Nothing but the marker (and its decoration) left: the model found no answer.
    if not any(character.isalnum() for character in answer):
        logger.info("rag.no_answer", extra={"prompt_id": _SYSTEM_PROMPT.id, "context_chunks": len(chunks)})
        return None
    return RagAnswer(answer=answer, sources=tuple(chunks))


def _format_context(chunks: Sequence[RetrievedChunk]) -> str:
    # Numbered so the model can refer to an excerpt, titled so it knows which
    # document a claim came from.
    return "\n\n".join(
        f"[{index}] {chunk.document_title}\n{chunk.chunk_text}"
        for index, chunk in enumerate(chunks, start=1)
    )
