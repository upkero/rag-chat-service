FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --no-dev --no-install-project --frozen

# Opt-in: local sentence-transformers models (CPU torch, ~1 GB more image).
# Off by default, so a plain `docker compose up` stays small and keyless.
ARG LOCAL_MODELS=false
RUN if [ "$LOCAL_MODELS" = "true" ]; then \
      uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch && \
      uv pip install --python .venv/bin/python sentence-transformers; \
    fi

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app" \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/models

RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home appuser && \
    mkdir -p /app /models && \
    chown appuser:appgroup /app /models

WORKDIR /app

COPY --from=builder --chown=appuser:appgroup /app/.venv .venv
COPY --chown=appuser:appgroup src/ src/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')"

CMD ["uvicorn", "src.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--no-access-log"]
