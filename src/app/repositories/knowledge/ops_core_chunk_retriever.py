import logging
from collections.abc import Sequence
from typing import Any

import httpx

from src.app.contracts.rag.retrieved_chunk import RetrievedChunk
from src.app.core.resilience import retry_async
from src.app.exceptions.retrieval import RetrievalThrottledError, RetrievalUnavailableError
from src.app.interfaces.knowledge.chunk_retriever import ChunkRetriever

logger = logging.getLogger(__name__)

_SEARCH_PATH = "/api/v1/documents/search"
_HEALTH_PATH = "/health/ready"
# Transient failures worth retrying: network/timeout errors + upstream 5xx.
_RETRYABLE: tuple[type[Exception], ...] = (httpx.TransportError, httpx.HTTPStatusError)
# Retryable status codes beyond 5xx: rate limiting and temporary unavailability.
# ops-core-api caps /documents/search at 20 req/min, and this service calls it once
# per request, so 429 is a normal Tuesday here rather than an exceptional event.
_RETRYABLE_STATUS = frozenset({429, 503})
# What we tell a client to wait when the upstream throttled us but named no delay.
_DEFAULT_RETRY_AFTER = "1"


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
        max_attempts: int,
        max_wait_seconds: float = 2.0,
    ) -> None:
        self._client = client
        self._attempts = max_attempts
        # Caps the backoff curve. Tests set it to 0 to keep the retry cases instant;
        # an upstream Retry-After still overrides it, which is the point of the header.
        self._max_wait_seconds = max_wait_seconds

    async def search(self, query: str, top_k: int) -> Sequence[RetrievedChunk]:
        payload = {"query": query, "top_k": top_k}

        # The retry policy is applied here rather than around the whole method so
        # that only the network call repeats; parsing the response is deterministic
        # and retrying it would just fail the same way at the same cost.
        @retry_async(
            attempts=self._attempts,
            retry_on=_RETRYABLE,
            max_wait_seconds=self._max_wait_seconds,
        )
        async def _call() -> httpx.Response:
            response = await self._client.post(_SEARCH_PATH, json=payload)
            # Retry server-side failures and being throttled: raise a (retryable)
            # HTTPStatusError for 5xx and for 429/503. Other 4xx are client
            # problems where retrying is pointless, so they fall through below.
            if response.status_code >= 500 or response.status_code in _RETRYABLE_STATUS:
                response.raise_for_status()
            return response

        try:
            response = await _call()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                # Still throttled after every attempt. Pass the upstream's own
                # Retry-After through when it gave one — it knows its window
                # better than any constant we could pick here.
                raise RetrievalThrottledError(
                    headers={
                        "Retry-After": exc.response.headers.get("Retry-After", _DEFAULT_RETRY_AFTER)
                    }
                ) from exc
            raise RetrievalUnavailableError(
                "Failed to reach ops-core-api document search."
            ) from exc
        except httpx.HTTPError as exc:
            # Any transport error surviving the retries → typed 502.
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
        # Exactly 200, not "not a server error": ops-core-api answers its readiness
        # probe with 503 when its database is down, which is precisely the state in
        # which document search cannot work. Anything looser reports us ready at the
        # one moment we certainly are not.
        return response.status_code == 200

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
