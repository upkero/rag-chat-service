from src.app.core.rate_limit import SlidingWindowRateLimiter


def test_allows_up_to_limit_then_blocks() -> None:
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60.0)
    assert limiter.allow("ip") is True
    assert limiter.allow("ip") is True
    assert limiter.allow("ip") is False


def test_separate_keys_are_independent() -> None:
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60.0)
    assert limiter.allow("a") is True
    assert limiter.allow("b") is True
    assert limiter.allow("a") is False


def test_window_evicts_old_hits(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    clock = {"now": 1000.0}
    monkeypatch.setattr("src.app.core.rate_limit.time.monotonic", lambda: clock["now"])

    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=10.0)
    assert limiter.allow("ip") is True
    assert limiter.allow("ip") is False

    clock["now"] += 11  # advance past the window
    assert limiter.allow("ip") is True
