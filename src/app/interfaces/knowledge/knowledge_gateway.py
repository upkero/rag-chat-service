from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.app.contracts.rag.retrieved_chunk import RetrievedChunk


class KnowledgeGateway(ABC):
    """Port onto the knowledge base an answer is grounded in.

    The service depends on this interface, not on any concrete backend
    (Dependency Inversion). The port is shaped like data access on purpose — ask
    for chunks, get chunks — so an HTTP-backed adapter (ops-core-api) and a
    hypothetical direct-DB one are interchangeable and RagService can never tell
    which one it is holding (Open/Closed).

    It is named a *gateway* rather than a repository because what sits behind it
    is somebody else's service across a network, not a database this service
    owns: it can be slow, throttle us, or be down while our own process is
    perfectly healthy. That is why the port has a ``ping`` at all.
    """

    @abstractmethod
    async def search(self, query: str, top_k: int) -> Sequence[RetrievedChunk]:
        """Return up to ``top_k`` chunks most relevant to ``query``, best first."""

    @abstractmethod
    async def ping(self) -> bool:
        """Return True if the knowledge backend is reachable (for readiness)."""

    @abstractmethod
    async def close(self) -> None:
        """Release any underlying resources (HTTP client, DB session, ...)."""
