from collections.abc import Sequence
from dataclasses import replace

import httpx

from src.app.contracts.rag.retrieved_chunk import RetrievedChunk
from src.app.interfaces.rerank.reranker import Reranker


class HttpReranker(Reranker):
    """Adapter for a Cohere-style ``/rerank`` endpoint (OpenRouter, Cohere, Voyage, ...)."""

    def __init__(self, *, client: httpx.AsyncClient, model: str) -> None:
        self._client = client
        self._model = model

    async def rerank(self, query: str, chunks: Sequence[RetrievedChunk], top_n: int) -> Sequence[RetrievedChunk]:
        if not chunks:
            return ()
        response = await self._client.post(
            "/rerank",
            json={
                "model": self._model,
                "query": query,
                "documents": [chunk.chunk_text for chunk in chunks],
                "top_n": min(top_n, len(chunks)),
            },
        )
        response.raise_for_status()
        results = response.json()["results"]
        return [replace(chunks[item["index"]], score=float(item["relevance_score"])) for item in results]

    async def close(self) -> None:
        await self._client.aclose()
