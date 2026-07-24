from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.app.contracts.rag.retrieved_chunk import RetrievedChunk


class ChunkRetriever(ABC):
    """Repository-style abstraction over document-chunk retrieval.

    The service depends on this interface, not on any concrete backend
    (Dependency Inversion). It deliberately looks like a data-access repository
    so an HTTP-backed adapter (ops-core-api) and a hypothetical direct-DB
    repository are interchangeable — swapping one for the other never touches
    RagService (Open/Closed).
    """

    @abstractmethod
    async def search(self, query: str, top_k: int) -> Sequence[RetrievedChunk]:
        """Return up to ``top_k`` chunks most relevant to ``query``, best first."""

    @abstractmethod
    async def close(self) -> None:
        """Release any underlying resources (HTTP client, DB session, ...)."""
