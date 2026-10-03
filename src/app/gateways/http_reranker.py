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
        for item in results:
            # A negative index would silently pick a chunk from the end. A broken answer
            # is an error, so the service falls back to the embedding order.
            if not 0 <= item["index"] < len(chunks):
                raise ValueError(f"rerank returned index {item['index']} for {len(chunks)} documents")
        return [replace(chunks[item["index"]], score=float(item["relevance_score"])) for item in results]

    async def close(self) -> None:
        await self._client.aclose()
