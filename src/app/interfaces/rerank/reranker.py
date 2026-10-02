from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.app.contracts.rag.retrieved_chunk import RetrievedChunk


class Reranker(ABC):
    """Port for a cross-encoder that re-scores retrieved chunks against the query.

    Embedding search compares two vectors that never saw each other; a
    cross-encoder reads the query and each chunk together, so its score is a far
    better answer to "does this passage actually answer the question".
    """

    @abstractmethod
    async def rerank(self, query: str, chunks: Sequence[RetrievedChunk], top_n: int) -> Sequence[RetrievedChunk]:
        """Return at most ``top_n`` chunks, best first, with ``score`` replaced by the rerank score."""

    @abstractmethod
    async def close(self) -> None:
        """Release the HTTP client."""
