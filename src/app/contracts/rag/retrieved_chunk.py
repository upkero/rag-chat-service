from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A single document chunk returned by retrieval (internal DTO).

    Framework-free currency between the retriever and the service. It mirrors the
    subset of ops-core-api's ChunkMatch we actually use, decoupled from its wire
    schema. ``score`` is cosine similarity in [0, 1] — higher = more relevant.
    """

    document_title: str
    chunk_index: int
    chunk_text: str
    score: float
