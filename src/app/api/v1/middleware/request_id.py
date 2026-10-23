import logging
import re
import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from opentelemetry import trace

from src.app.core.request_id import generate_request_id, set_request_id

logger = logging.getLogger(__name__)

_HEADER = "X-Request-ID"
_SKIP_PATHS = frozenset({"/metrics", "/health/live", "/health/ready"})
# The id is echoed back, forwarded upstream and written into every log line, so
# only a plain token is taken from the caller; anything else gets a fresh one.
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,128}")


def register_request_id_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_id_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)
        incoming = request.headers.get(_HEADER, "")
        request_id = incoming if _VALID_REQUEST_ID.fullmatch(incoming) else generate_request_id()
        set_request_id(request_id)
        # On the server span too, so a trace is found by the id a log line shows.
        trace.get_current_span().set_attribute("request_id", request_id)
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            # The one line per request (uvicorn's access log is off): the route's template,
            # never the raw path, and no query string, body or client address.
            # The server span's route, the same one the metrics carry: scope["route"] holds
            # only the path relative to its router ("/ask", not "/api/v1/ask").
            route = (getattr(trace.get_current_span(), "attributes", None) or {}).get("http.route")
            logger.info(
                "http.request",
                extra={
                    "method": request.method,
                    "route": route or "<unmatched>",
                    "status": status,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
        response.headers[_HEADER] = request_id
        return response
