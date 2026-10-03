import math
import time
from collections import defaultdict, deque


class SlidingWindowRateLimiter:
    """In-memory sliding-window rate limiter, keyed by an arbitrary client id.

    NOTE: per-process and in-memory — correct for a single instance / demo.
    For multi-instance production move the counter to Redis (or use slowapi) so
    the limit is shared across workers; the ``allow`` contract stays the same.
    """

    def __init__(self, *, max_requests: int, window_seconds: float) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        """Return True if a request from ``key`` is within the limit, else False."""
        now = time.monotonic()
        cutoff = now - self._window
        hits = self._hits[key]
        while hits and hits[0] < cutoff:
            hits.popleft()
        if len(hits) >= self._max:
            return False
        hits.append(now)
        return True

    def retry_after(self, key: str) -> int:
        """Whole seconds (at least 1) until ``key``'s oldest hit leaves the window."""
        hits = self._hits.get(key)
        if not hits:
            return 1
        return max(1, math.ceil(hits[0] + self._window - time.monotonic()))
