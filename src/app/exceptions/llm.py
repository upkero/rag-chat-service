from src.app.exceptions.base import BaseAppException


class LLMError(BaseAppException):
    """Base exception for LLM client operations."""

    error_code = "llm_error"
    default_detail = "LLM operation failed."


class LLMConfigurationError(LLMError):
    """Raised when LLM client configuration is invalid."""

    error_code = "llm_configuration_error"


class LLMInputError(LLMError, ValueError):
    """Raised when LLM input payload is invalid."""

    status_code = 422
    error_code = "llm_input_error"


class LLMGenerationError(LLMError):
    """Raised when provider generation fails."""

    error_code = "llm_generation_error"


class GenerationUnavailableError(LLMError):
    """Raised when the LLM provider is unreachable or failing.

    Covers rate limits, timeouts and network errors during the generate phase.
    Surfaces as a 502 (upstream failure) so the router returns a clean error
    envelope instead of leaking a raw provider error — the generate-phase
    counterpart of RetrievalUnavailableError.
    """

    status_code = 502
    error_code = "generation_unavailable"
    default_detail = "Answer generation service is unavailable."
