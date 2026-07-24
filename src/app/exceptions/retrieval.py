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
