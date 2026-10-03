from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Request

from src.app.core.rate_limit import SlidingWindowRateLimiter
from src.app.core.settings.rate_limit import get_rate_limit_settings
from src.app.exceptions.rate_limit import TooManyRequestsError


@lru_cache(maxsize=1)
def get_rate_limiter() -> SlidingWindowRateLimiter:
    # Process-wide singleton so the window is shared across requests. Exposed as a
    # dependency (not a bare import) so tests can override it via
    # app.dependency_overrides with a tighter limiter.
    settings = get_rate_limit_settings()
    return SlidingWindowRateLimiter(
        max_requests=settings.max_requests,
        window_seconds=settings.window_seconds,
    )


def enforce_rate_limit(
    request: Request,
    limiter: Annotated[SlidingWindowRateLimiter, Depends(get_rate_limiter)],
) -> None:
    client = request.client.host if request.client else "unknown"
    if not limiter.allow(client):
        raise TooManyRequestsError(headers={"Retry-After": str(limiter.retry_after(client))})
