from src.app.exceptions.base import BaseAppException


class TooManyRequestsError(BaseAppException):
    """Raised when a client exceeds the allowed request rate."""

    status_code = 429
    error_code = "rate_limited"
    default_detail = "Rate limit exceeded. Try again later."
