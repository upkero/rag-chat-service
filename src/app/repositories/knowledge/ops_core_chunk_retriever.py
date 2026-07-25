import logging
from collections.abc import Sequence
from typing import Any

import httpx

from src.app.contracts.rag.retrieved_chunk import RetrievedChunk
from src.app.core.retry import retry_async
from src.app.exceptions.retrieval import RetrievalUnavailableError
from src.app.interfaces.knowledge.chunk_retriever import ChunkRetriever

logger = logging.getLogger(__name__)

_SEARCH_PATH = "/api/v1/documents/search"
_HEALTH_PATH = "/health"
# Transient failures worth retrying: network/timeout errors + upstream 5xx.
_RETRYABLE: tuple[type[Exception], ...] = (httpx.TransportError, httpx.HTTPStatusError)


class OpsCoreChunkRetriever(ChunkRetriever):
    """Adapter: exposes ops-core-api's HTTP document search behind ChunkRetriever.

    Adapter pattern — it wraps an httpx client and translates ops-core-api's wire
    schema into our RetrievedChunk contracts, so RagService sees exactly the same
    interface a direct-DB repository would expose. Every HTTP concern (URL, auth
    header, JSON shape, retries, error mapping) is contained here and never leaks
    upward.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        max_retries: int,
        retry_base_delay: float = 0.2,
    ) -> None:
        self._client = client
        self._attempts = max_retries + 1
        self._retry_base_delay = retry_base_delay

    async def search(self, query: str, top_k: int) -> Sequence[RetrievedChunk]:
        payload = {"query": query, "top_k": top_k}

        async def _call() -> httpx.Response:
            response = await self._client.post(_SEARCH_PATH, json=payload)
            # Retry only server-side failures: raise a (retryable) HTTPStatusError
            # for 5xx; 4xx is a client problem and retrying it is pointless.
            if response.status_code >= 500:
                response.raise_for_status()
            return response

        try:
            response = await retry_async(
                _call,
                attempts=self._attempts,
                base_delay=self._retry_base_delay,
                retry_on=_RETRYABLE,
            )
        except httpx.HTTPError as exc:
            # Any transport/5xx error surviving the retries → typed 502.
            raise RetrievalUnavailableError(
                "Failed to reach ops-core-api document search."
            ) from exc

        if response.is_error:
            # Non-retryable upstream response (e.g. 401 bad key, 422) → typed 502.
            raise RetrievalUnavailableError(
                f"ops-core-api returned {response.status_code} for document search."
            )

        try:
            chunks = self._to_chunks(response.json())
        except (KeyError, TypeError, ValueError) as exc:
            # Malformed / unexpected payload (bad JSON, missing or wrong-typed
            # fields) → typed 502 instead of an unhandled 500.
            raise RetrievalUnavailableError(
                "ops-core-api returned an unexpected response schema."
            ) from exc

        logger.info(
            "ops_core.search",
            extra={"top_k": top_k, "matches_returned": len(chunks)},
        )
        return chunks

    async def ping(self) -> bool:
        try:
            response = await self._client.get(_HEALTH_PATH)
        except httpx.HTTPError:
            return False
        return response.status_code < 500

    async def close(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _to_chunks(body: dict[str, Any]) -> list[RetrievedChunk]:
        # Boundary mapping: ops-core-api wire schema -> internal contracts.
        matches = body.get("matches", [])
        return [
            RetrievedChunk(
                document_title=match["document_title"],
                chunk_index=match["chunk_index"],
                chunk_text=match["chunk_text"],
                score=match["score"],
            )
            for match in matches
        ]
