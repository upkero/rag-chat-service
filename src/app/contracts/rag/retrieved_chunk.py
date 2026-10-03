from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A single document chunk returned by retrieval (internal DTO).

    Framework-free currency between the retriever and the service. It mirrors the
    subset of ops-core-api's ChunkMatch we actually use, decoupled from its wire
    schema.

    ``score`` is in [0, 1], higher = more relevant, but its scale depends on who set it:

    * without a reranker it is ops-core-api's cosine similarity, gated by
      ``RAG_SCORE_THRESHOLD`` (default 0.5);
    * with a reranker it is the cross-encoder's relevance score, which replaces the
      cosine value and is gated by ``RERANK_MIN_SCORE`` (default 0.08). It is not a
      similarity: relevant passages typically land at 0.10–0.70 on Cohere rerank-v3.5.

    Only the scale differs; a score is only ever compared with the gate of its own mode.
    """

    document_title: str
    chunk_index: int
    chunk_text: str
    score: float
