import pytest

from src.app.core.retry import retry_async


async def test_retry_succeeds_after_transient_failures() -> None:
    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("transient")
        return "ok"

    result = await retry_async(flaky, attempts=3, base_delay=0.0, retry_on=(ValueError,))

    assert result == "ok"
    assert calls["n"] == 3


async def test_retry_gives_up_and_reraises_last() -> None:
    calls = {"n": 0}

    async def always_fails() -> str:
        calls["n"] += 1
        raise ValueError(f"fail-{calls['n']}")

    with pytest.raises(ValueError, match="fail-2"):
        await retry_async(always_fails, attempts=2, base_delay=0.0, retry_on=(ValueError,))

    assert calls["n"] == 2


async def test_retry_does_not_catch_unlisted_exceptions() -> None:
    async def raises_type_error() -> str:
        raise TypeError("nope")

    with pytest.raises(TypeError):
        await retry_async(raises_type_error, attempts=3, base_delay=0.0, retry_on=(ValueError,))
