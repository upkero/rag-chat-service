from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.app.api.v1.exception_handlers import register_exception_handlers
from src.app.api.v1.middleware.request_id import register_request_id_middleware
from src.app.api.v1.router import api_router
from src.app.api.v1.routers.health import router as health_router
from src.app.bootstrap.container import ApplicationContainer
from src.app.core.logging import setup_logging
from src.app.core.settings.app import get_app_settings
from src.app.core.settings.logging import get_logging_settings
from src.app.core.settings.rate_limit import get_rate_limit_settings
from src.app.core.settings.security import get_security_settings
from src.app.core.telemetry import instrument_app, setup_tracing


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    container = ApplicationContainer()
    app.state.container = container
    try:
        # The container is lazy; build the graph the requests use now, so a
        # misconfiguration fails the boot instead of the first request.
        _ = container.rag_service, get_security_settings(), get_rate_limit_settings()
        yield
    finally:
        await container.close()


def create_app() -> FastAPI:
    setup_logging(get_logging_settings())
    setup_tracing("rag-chat-service", "rag-chat-service")

    app = FastAPI(title="RAG Chat Service", version="0.1.0", lifespan=lifespan)

    register_request_id_middleware(app)
    # Added last, so outermost (C10): a 429 or a preflight is answered with CORS headers
    # and an OPTIONS request never reaches anything that spends quota.
    settings = get_app_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        # Explicitly False: the caller authenticates with a header, never a cookie,
        # so there is nothing for the browser to send along. Deriving this from the
        # origin list only produced a True that no request could ever use.
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        # Without this browser JavaScript cannot read the two headers a failed call is
        # diagnosed and retried by.
        expose_headers=["Retry-After", "X-Request-ID"],
    )

    register_exception_handlers(app)
    # Last, so the server span is the outermost and also wraps the CORS and 429 answers.
    instrument_app(app)

    app.include_router(health_router)
    app.include_router(api_router)

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
