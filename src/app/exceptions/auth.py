from src.app.exceptions.base import BaseAppException


class UnauthorizedError(BaseAppException):
    """Raised when a request is missing or presents an invalid API key."""

    status_code = 401
    error_code = "unauthorized"
    default_detail = "Missing or invalid API key."
