import asyncio
import time
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

from openai import OpenAIError
from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind

from src.app.contracts.llm.llm_message import LLMMessage
from src.app.contracts.llm.llm_response import LLMResponse
from src.app.core import metrics
from src.app.core.settings.llm import LLMSettings
from src.app.exceptions.llm import LLMGenerationError, LLMInputError
from src.app.interfaces.llm.llm_client import LLMClient

_USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")

_tracer = trace.get_tracer(__name__)

class OpenAICompatibleLLMClient(LLMClient):
    """Concrete LLMClient over any OpenAI-compatible endpoint (OpenAI, Ollama, ...).

    Maps raw SDK errors to typed domain exceptions and normalises the SDK's
    response object into our LLMResponse contract, so nothing above this layer
    sees the OpenAI SDK.
    """

    def __init__(
        self,
        *,
        settings: LLMSettings,
        client: Any,
    ) -> None:
        self._settings = settings
        self._client = client
        self._server = urlparse(settings.base_url or "").hostname or ""
        # The one provider with a request flag we use: it makes the reply carry the exact
        # cost of the call. Sent to any other endpoint it is an unknown parameter.
        self._is_openrouter = self._server == "openrouter.ai" or self._server.endswith(".openrouter.ai")
        self._ping_result: bool | None = None
        self._ping_expires = 0.0
        self._ping_lock = asyncio.Lock()
        # Readiness is polled every ~10 s per caller and each probe is a real request
        # to a provider whose latency swings from 0.3 s to several, so an answer is
        # reused: long when healthy, short when not, so a recovery shows on the next poll.

    @property
    def model_name(self) -> str:
        return self._settings.model

    @property
    def provider_name(self) -> str:
        return "openrouter" if self._is_openrouter else self._settings.provider

    async def complete(self, messages: Sequence[LLMMessage]) -> LLMResponse:
        if not messages:
            raise LLMInputError("LLM messages must not be empty.")

        attributes = {
            "gen_ai.provider.name": self.provider_name,
            "gen_ai.request.model": self.model_name,
        }
        with _tracer.start_as_current_span(
            f"chat {self.model_name}",
            kind=SpanKind.CLIENT,
            attributes={**attributes, "gen_ai.operation.name": "chat", "server.address": self._server},
        ) as span:
            started = time.perf_counter()
            error_type: str | None = None
            try:
                try:
                    completion = await self._client.chat.completions.create(
                        **self._completion_params(messages),
                    )
                except OpenAIError as exc:
                    raise LLMGenerationError("LLM provider request failed.") from exc
                response = self._build_response(completion)
            except Exception as exc:
                # The cause's class when there is one ("RateLimitError"), not our wrapper's:
                # that is what tells a throttled provider from a broken one.
                error_type = type(exc.__cause__ or exc).__name__
                span.set_attribute("error.type", error_type)
                raise
            finally:
                metrics.operation_duration.record(
                    time.perf_counter() - started,
                    {**attributes, **({"error.type": error_type} if error_type else {})},
                )
            self._record_response(span, completion, response, attributes)
            return response

    async def ping(self) -> bool:
        if self._ping_result is not None and time.monotonic() < self._ping_expires:
            return self._ping_result
        # One probe at a time: pollers arriving together share its answer instead
        # of each sending their own request.
        async with self._ping_lock:
            if self._ping_result is not None and time.monotonic() < self._ping_expires:
                return self._ping_result
            try:
                await self._client.models.list()
                healthy = True
            except OpenAIError:
                healthy = False
            self._ping_result = healthy
            self._ping_expires = time.monotonic() + (
                self._settings.ping_ok_ttl_seconds if healthy else self._settings.ping_failed_ttl_seconds
            )
            return healthy

    async def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            await close()

    def _completion_params(self, messages: Sequence[LLMMessage]) -> dict[str, object]:
        params: dict[str, object] = {
            "model": self._settings.model,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in messages
            ],
        }
        if self._settings.temperature is not None:
            params["temperature"] = self._settings.temperature
        if self._settings.max_tokens is not None:
            params["max_tokens"] = self._settings.max_tokens
        if self._settings.reasoning_effort is not None:
            params["reasoning_effort"] = self._settings.reasoning_effort
        if self._is_openrouter:
            params["extra_body"] = {"usage": {"include": True}}
        return params

    def _record_response(
        self, span: Span, completion: Any, response: LLMResponse, attributes: Mapping[str, str]
    ) -> None:
        """What the reply says about itself, onto the span and the metrics.

        Token counts, finish reason and cost only: never the prompt or the reply.
        """
        metadata = response.metadata or {}
        span.set_attribute("gen_ai.response.model", response.model or self.model_name)
        if "id" in metadata:
            span.set_attribute("gen_ai.response.id", str(metadata["id"]))
        if response.finish_reason is not None:
            span.set_attribute("gen_ai.response.finish_reasons", [response.finish_reason])
            metrics.finish_reasons.add(1, {"finish_reason": response.finish_reason})
        usage = response.usage or {}
        for token_type, field, attribute in (
            ("input", "prompt_tokens", "gen_ai.usage.input_tokens"),
            ("output", "completion_tokens", "gen_ai.usage.output_tokens"),
        ):
            if field in usage:
                span.set_attribute(attribute, usage[field])
                metrics.token_usage.record(usage[field], {**attributes, "gen_ai.token.type": token_type})
        # Only OpenRouter reports it, and the SDK keeps unknown usage fields as attributes.
        cost = getattr(getattr(completion, "usage", None), "cost", None)
        if isinstance(cost, int | float):
            span.set_attribute("llm.cost", float(cost))
            metrics.llm_cost.add(
                float(cost), {"provider": self.provider_name, "model": self.model_name}
            )

    def _build_response(self, completion: Any) -> LLMResponse:
        choice = self._first_choice(completion)
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        if content is None:
            raise LLMGenerationError("LLM provider returned empty message content.")

        return LLMResponse(
            content=content,
            model=getattr(completion, "model", None) or self.model_name,
            finish_reason=getattr(choice, "finish_reason", None),
            usage=self._usage_to_mapping(getattr(completion, "usage", None)),
            metadata=self._metadata(completion),
        )

    @staticmethod
    def _first_choice(completion: Any) -> Any:
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise LLMGenerationError("LLM provider returned no choices.")
        return choices[0]

    @staticmethod
    def _usage_to_mapping(usage: Any) -> Mapping[str, int] | None:
        if usage is None:
            return None
        result: dict[str, int] = {}
        for field in _USAGE_FIELDS:
            value = getattr(usage, field, None)
            if isinstance(value, int):
                result[field] = value
        return result or None

    @staticmethod
    def _metadata(completion: Any) -> Mapping[str, object]:
        metadata: dict[str, object] = {}
        completion_id = getattr(completion, "id", None)
        created = getattr(completion, "created", None)
        if completion_id is not None:
            metadata["id"] = completion_id
        if created is not None:
            metadata["created"] = created
        return metadata
