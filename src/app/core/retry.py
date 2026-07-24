import asyncio
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    base_delay: float,
    retry_on: tuple[type[Exception], ...],
) -> T:
    """Run ``operation`` with exponential-backoff retries — a reusable resilience util.

    Extracted here (DRY) so every outbound call — HTTP to ops-core-api, and any
    future external call — shares one retry/backoff policy instead of hand-rolling
    loops. Only ``retry_on`` exception types are retried; anything else propagates
    immediately. After ``attempts`` failures the last exception is re-raised.

    Sleeps ``base_delay * 2**n`` between attempts (n = 0-based retry index).
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return await operation()
        except retry_on as exc:
            last_exc = exc
            if attempt + 1 >= attempts:
                break
            delay = base_delay * (2**attempt)
            logger.warning(
                "outbound call failed; retrying",
                extra={
                    "attempt": attempt + 1,
                    "max_attempts": attempts,
                    "delay_seconds": round(delay, 3),
                    "error": str(exc),
                },
            )
            await asyncio.sleep(delay)

    # Only reachable after at least one failed attempt, so last_exc is set.
    assert last_exc is not None
    raise last_exc
