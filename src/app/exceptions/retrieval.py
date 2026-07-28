from src.app.exceptions.base import BaseAppException


class RetrievalError(BaseAppException):
    """Base exception for document-retrieval operations."""

    error_code = "retrieval_error"
    default_detail = "Document retrieval failed."


class RetrievalUnavailableError(RetrievalError):
    """Raised when ops-core-api is unreachable or returns an error.

    Surfaces as a 502 (upstream failure) rather than a 500, so a flaky backend
    reads as "upstream unavailable" instead of an internal crash.
    """

    status_code = 502
    error_code = "retrieval_unavailable"
    default_detail = "Document search service is unavailable."


class RetrievalThrottledError(RetrievalError):
    """Raised when ops-core-api is still rate-limiting us after every retry.

    Deliberately not a RetrievalUnavailableError: "wait and try again" and "the
    upstream is broken" are different answers, and collapsing the first into the
    second throws away the only actionable thing the caller was told. Surfaces as
    a 429 carrying our own Retry-After, so the client learns how long to wait
    instead of seeing a 502 it can only guess at.
    """

    status_code = 429
    error_code = "rate_limit_exceeded"
    default_detail = "Document search is rate limited upstream. Try again later."
