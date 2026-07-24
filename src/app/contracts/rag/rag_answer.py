from collections.abc import Sequence
from dataclasses import dataclass

from src.app.contracts.rag.retrieved_chunk import RetrievedChunk


@dataclass(frozen=True, slots=True)
class RagAnswer:
    """Result of a RAG ask: the answer plus the sources it was grounded on.

    ``from_llm`` is False when the confidence gate short-circuited to the canned
    fallback (the LLM was never called) — useful for logging and metrics.
    """

    answer: str
    sources: Sequence[RetrievedChunk]
    from_llm: bool
