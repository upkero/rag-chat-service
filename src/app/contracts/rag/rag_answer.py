from collections.abc import Sequence
from dataclasses import dataclass

from src.app.contracts.rag.retrieved_chunk import RetrievedChunk


@dataclass(frozen=True, slots=True)
class RagAnswer:
    """Result of a RAG ask: the answer plus the sources it was grounded on.

    An empty ``sources`` means a refusal: the canned message, not a model answer. It
    happens two ways, and only the first skips the LLM: the confidence gate short-circuits
    (the LLM is never called), or the chunks passed the gate but the model replied with
    ``NO_ANSWER`` (the LLM was called). There is no separate flag for which one; the
    ``rag.answers`` outcome (``low_confidence_gate`` or ``model_no_answer``) and the gate's
    ``llm_skipped`` log line say it for anyone counting.
    """

    answer: str
    sources: Sequence[RetrievedChunk]
