from collections.abc import Sequence
from dataclasses import dataclass

from src.app.contracts.rag.retrieved_chunk import RetrievedChunk


@dataclass(frozen=True, slots=True)
class RagAnswer:
    """Result of a RAG ask: the answer plus the sources it was grounded on.

    An empty ``sources`` means the confidence gate short-circuited to the canned
    fallback and the LLM was never called. There is no separate flag for that:
    one field that can contradict another is a bug waiting to be written, and the
    gate already logs ``llm_skipped`` for anyone counting.
    """

    answer: str
    sources: Sequence[RetrievedChunk]
