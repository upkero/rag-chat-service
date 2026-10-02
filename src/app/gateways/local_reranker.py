import asyncio
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from src.app.contracts.rag.retrieved_chunk import RetrievedChunk
from src.app.interfaces.rerank.reranker import Reranker


class LocalReranker(Reranker):
    """Adapter: a sentence-transformers CrossEncoder (e.g. BAAI/bge-reranker-v2-m3) in this process.

    Scores come out of a sigmoid, so they are in [0, 1] like the HTTP reranker's.
    The model loads at construction (startup) so a missing package or a bad model
    name fails the boot, not the first question. Scoring is CPU-bound and runs in
    a worker thread.
    """

    def __init__(self, *, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "RERANK_PROVIDER=local needs sentence-transformers; build the image with LOCAL_MODELS=true."
            ) from exc
        self._model: Any = CrossEncoder(model_name, device="cpu")

    async def rerank(self, query: str, chunks: Sequence[RetrievedChunk], top_n: int) -> Sequence[RetrievedChunk]:
        if not chunks:
            return ()
        pairs = [(query, chunk.chunk_text) for chunk in chunks]
        scores = await asyncio.to_thread(self._model.predict, pairs)
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [replace(chunk, score=float(score)) for chunk, score in ranked[:top_n]]

    async def close(self) -> None:
        return None
