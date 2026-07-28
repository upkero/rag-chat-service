"""Pin the error codes this service shares with ops-core-api.

`tests/fixtures/error-codes.json` is a manual copy of ops-core-api's generated
`docs/error-codes.json`. Refreshing it stays manual, but a *test* now fails when
it goes stale, instead of a client discovering the rename in production.

Unlike the other consumers this service has no `_ERROR_CODES` map: its gateway
switches on the upstream's HTTP status, never on its error code, so there is
nothing here that silently mis-parses a renamed one. What it does share is the
vocabulary itself — a caller handling `invalid_api_key` from ops-core-api must
get the same string from us — so that is what is asserted.
"""

import json
from pathlib import Path

from src.app.exceptions.auth import UnauthorizedError
from src.app.exceptions.base import BaseAppException
from src.app.exceptions.rate_limit import TooManyRequestsError
from src.app.exceptions.retrieval import RetrievalThrottledError

_CONTRACT = json.loads((Path(__file__).parents[1] / "fixtures" / "error-codes.json").read_text("utf-8"))

# Our exceptions whose code is part of the shared vocabulary, not our own.
_SHARED: tuple[type[BaseAppException], ...] = (
    UnauthorizedError,
    TooManyRequestsError,
    RetrievalThrottledError,
)


def test_shared_error_codes_exist_in_the_ops_core_contract() -> None:
    for exception in _SHARED:
        assert exception.error_code in _CONTRACT, (
            f"{exception.__name__} emits {exception.error_code!r}, which ops-core-api no longer "
            f"publishes. Either the code was renamed upstream or the fixture is stale."
        )


def test_shared_error_codes_keep_the_upstream_status() -> None:
    """Same code, same status — or the code means two things to one client."""
    for exception in _SHARED:
        assert exception.status_code == _CONTRACT[exception.error_code]["status_code"]
