# RAG Chat Service

[![CI](https://github.com/upkero/rag-chat-service/actions/workflows/ci.yml/badge.svg)](https://github.com/upkero/rag-chat-service/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

A thin Retrieval-Augmented-Generation chatbot built **on top of `ops-core-api`**. It
stores no documents of its own: on each question it (1) calls `ops-core-api`'s
`POST /api/v1/documents/search` to retrieve the most relevant document chunks, then
(2) asks an LLM to answer **strictly from that retrieved context**.

A hard business rule gates step 2: if the best chunk's similarity `score` is below a
configurable threshold (`RAG_SCORE_THRESHOLD`, default `0.5`), the service returns a
canned *"can't answer precisely, ask an administrator"* message **without calling the
LLM at all** — cost control plus honesty (no hallucinating on weak retrieval).

## Architecture

Layered architecture with a strict inward dependency rule (outer layers depend on
inner, never the reverse). Because this service persists nothing, it has **no database
layer** and therefore no `repositories/`: what would be one is a `gateways/` **Adapter**
onto `ops-core-api` behind the `KnowledgeGateway` port. A `repository` is a database
you own; a `gateway` is somebody else's service across a network — it can throttle you
or be down while your own process is perfectly healthy.

```
api/v1  (POST /api/v1/ask)
  └─ services    RagService: retrieve → confidence gate → generate
       ├─ interfaces/KnowledgeGateway ←impl─ gateways/OpsCoreKnowledgeGateway (HTTP Adapter)
       └─ interfaces/LLMClient        ←impl─ llm/OpenAICompatibleLLMClient (OpenAI / Ollama)
  llm/skills + prompts     one file per model job; every prompt is Markdown, in English
  messages                 text the user reads verbatim, per language
  core / exceptions / bootstrap (settings, JSON logging, DI container)
```

Patterns used (and commented in code): **Adapter** (HTTP client behind a data-access
shaped port), **Gateway** (`KnowledgeGateway`), **Factory**
(`create_ops_core_knowledge_gateway`, `create_llm_client`), **Template Method** (the
two-phase `RagService.ask`), and **Dependency Inversion** (the service depends only on
interfaces).

Model-facing prose lives in `src/app/prompts/*.md` and nowhere else — English, with the
reply language as a `{reply_language}` placeholder rather than a translated copy of the
file. Text a user reads verbatim is not a prompt and lives in `src/app/messages/`.

## Endpoint

`POST /api/v1/ask` — **requires** an `X-API-Key` header.

```jsonc
// request
{ "question": "how do I cancel an appointment" }

// response
{
  "answer": "You can cancel or reschedule free of charge up to twenty-four hours before the start time.",
  "sources": [
    { "document_title": "Cancellation and rescheduling policy", "chunk_index": 0, "score": 0.82 }
  ]
}
```

The examples throughout this README quote `ops-core-api`'s seeded demo documents, so
they reproduce against a backend seeded with `uv run python -m src.app.cli.seed`.

Health probes (open, no key): `GET /health/live`, `GET /health/ready` (readiness
checks **both** the LLM and ops-core-api, returns 503 if either is down).

## Security

- **API-key auth** on every `/api/v1/*` route via the `X-API-Key` header
  (constant-time compare; `SECURITY_API_KEY`, min length 16). Missing/invalid → 401.
- **Rate limiting** per client IP (sliding window; `RATE_LIMIT_*`, default **20/min**) →
  429 when exceeded. 20 is not arbitrary: `ops-core-api` caps `/documents/search` at
  20/min and this service calls it exactly once per question, so a higher number here
  would only relocate the 429 to a worse place to read it. In-memory/per-process — use
  Redis for a shared limit across instances.
- **CORS** closed by default (`CORS_ALLOWED_ORIGINS` empty, comma-separated) — set
  origins explicitly; a browser client cannot call the service until you do.
- Secrets only via env (`SECURITY_API_KEY`, `OPS_CORE_API_KEY`, `LLM_API_KEY`) — none
  are hard-coded. `.env.example` ships the placeholder `change-me-min-16-chars`, which
  is shared by all five portfolio services so `cp .env.example .env` gives a working
  demo; it is rotated in all five at once, and replaced everywhere before deployment.

## Run locally

Prerequisites: a running **`ops-core-api`** (seeded with documents) and an
**OpenAI-compatible LLM** (OpenAI, or a local Ollama — Ollama is OpenAI-compatible).

```bash
cp .env.example .env          # then edit OPS_CORE_* and LLM_* values
docker compose up --build     # service on http://localhost:8001
```

`docker-compose.yml` runs only this service (port `8001` to avoid `ops-core-api`'s
`8000`). Point `OPS_CORE_BASE_URL` / `LLM_BASE_URL` at your backends; when they run on
the host, use `http://host.docker.internal:<port>`.

Without Docker:

```bash
uv sync
uv run uvicorn src.main:app --reload --port 8001
```

## Example requests

```bash
# The key from .env (the shared placeholder is change-me-min-16-chars)
export SECURITY_API_KEY=change-me-min-16-chars   # or: set -a; . ./.env; set +a

# In-topic question → grounded answer with its source
curl -sX POST http://localhost:8001/api/v1/ask \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $SECURITY_API_KEY" \
  -d '{"question": "how do I cancel an appointment"}'

# Off-topic question → canned fallback, LLM never called (see logs: "llm_skipped": true)
curl -sX POST http://localhost:8001/api/v1/ask \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $SECURITY_API_KEY" \
  -d '{"question": "what is the weather on mars"}'
```

## Tests

```bash
uv run pytest --cov=src/app/services --cov-report=term-missing --cov-fail-under=60
uv run ruff check .
uv run mypy src
```

Unit tests mock the `KnowledgeGateway` and `LLMClient` interfaces and cover both DoD
scenarios (confident answer with source; below-threshold rejection with no LLM call),
plus generate-phase failure handling and language fidelity. Integration tests drive
`POST /api/v1/ask` through `httpx.ASGITransport` with the service dependency
overridden — no real `ops-core-api` or LLM is contacted.

`tests/fixtures/error-codes.json` is a copy of `ops-core-api`'s generated error-code
contract; a test fails if a code this service emits stops existing upstream.

## Configuration

All configuration is via environment variables (see [`.env.example`](.env.example)):
`SECURITY_API_KEY` (auth), `RATE_LIMIT_*` (throttle), `OPS_CORE_BASE_URL` /
`OPS_CORE_API_KEY` / `OPS_CORE_TIMEOUT_SECONDS` / `OPS_CORE_MAX_ATTEMPTS` (upstream
connection; attempts are the TOTAL including the first), `RAG_*` (business rules incl.
the score threshold), `LLM_*` (provider/model), `LOG_*`, and `CORS_ALLOWED_ORIGINS`.
Nothing is hard-coded; there is no database.

## License

[MIT](LICENSE) © 2026 upkero

---

# RAG Chat Service (RU)

Тонкий RAG-чатбот **поверх `ops-core-api`**. Сам документы не хранит: на каждый вопрос
он (1) вызывает `POST /api/v1/documents/search` в `ops-core-api` за релевантными
чанками, затем (2) просит LLM сформировать ответ **строго на основе полученного
контекста**.

Ключевое бизнес-правило гейтит шаг 2: если `score` лучшего чанка ниже порога
(`RAG_SCORE_THRESHOLD`, по умолчанию `0.5`) — сервис возвращает заранее заданный ответ
*«не могу ответить точно, уточните у администратора»* (см. `src/app/messages/`) и
**не обращается к LLM вообще** (экономия и честность — не даём модели домысливать при
слабом retrieval).

## Архитектура

Слоистая архитектура со строгим правилом однонаправленных зависимостей (внешние слои
зависят от внутренних, не наоборот). Так как сервис ничего не хранит, у него **нет слоя
БД** и, соответственно, нет `repositories/`: вместо него — `gateways/`, HTTP **Adapter**
к `ops-core-api` за портом `KnowledgeGateway`. Репозиторий — это своя база; шлюз — это
чужой сервис по сети, который может троттлить или лежать, пока свой процесс здоров.

Применённые паттерны (прокомментированы в коде): **Adapter**, **Gateway**,
**Factory**, **Template Method** (двухфазный `RagService.ask`) и **Dependency
Inversion** (сервис зависит только от интерфейсов).

Промпты живут в `src/app/prompts/*.md` — только там и только по-английски; язык ответа
задаётся плейсхолдером `{reply_language}`, а не переводом файла. Текст, который
пользователь читает дословно, — не промпт и лежит в `src/app/messages/`.

## Эндпоинт

`POST /api/v1/ask` — принимает `question`, возвращает `answer` и список `sources`
(`document_title` + `chunk_index` + `score`). **Требует заголовок `X-API-Key`.**

## Безопасность

- **API-key аутентификация** на всех `/api/v1/*` через заголовок `X-API-Key`
  (сравнение constant-time; `SECURITY_API_KEY`, минимум 16 символов) → 401 при
  отсутствии/неверном ключе.
- **Rate limiting** по IP клиента (скользящее окно; `RATE_LIMIT_*`, по умолчанию
  **20/мин**) → 429. Ровно 20, потому что `ops-core-api` ограничивает
  `/documents/search` теми же 20/мин, а сервис ходит туда один раз на вопрос: больший
  лимит здесь лишь переносит 429 туда, где его хуже читать. In-memory/per-process —
  для общего лимита между инстансами нужен Redis.
- **CORS** по умолчанию закрыт (`CORS_ALLOWED_ORIGINS` пуст, значения через запятую).
- Ключ-плейсхолдер `change-me-min-16-chars` из `.env.example` общий для всех пяти
  сервисов портфолио и ротируется во всех пяти сразу.
- `/health/*` открыты без ключа; readiness проверяет и LLM, и ops-core-api.

## Запуск

Нужны запущенный **`ops-core-api`** (с загруженными документами) и **OpenAI-совместимый
LLM** (OpenAI или локальная Ollama).

```bash
cp .env.example .env          # отредактируйте OPS_CORE_* и LLM_*
docker compose up --build     # сервис на http://localhost:8001
```

`docker-compose.yml` поднимает только этот сервис (порт `8001`, чтобы не конфликтовать
с `ops-core-api` на `8000`). Укажите `OPS_CORE_BASE_URL` / `LLM_BASE_URL` на свои
бэкенды; если они на хосте — используйте `http://host.docker.internal:<порт>`.

## Тесты

```bash
uv run pytest --cov=src/app/services --cov-report=term-missing --cov-fail-under=60
uv run ruff check . && uv run mypy src
```

Юнит-тесты мокают интерфейсы `KnowledgeGateway` и `LLMClient` и покрывают оба сценария
DoD (точный ответ с источником; отклонение по порогу без вызова LLM), а также обработку
сбоя генерации и языковую вер­ность. Интеграционные тесты дёргают `POST /api/v1/ask`
через `httpx.ASGITransport` с подменённым сервисом — реальные `ops-core-api` и LLM не
вызываются.
