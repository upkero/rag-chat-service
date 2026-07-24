# RAG Chat Service

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
inner, never the reverse) — see [`docs/architecture.md`](docs/architecture.md).
Because this service persists nothing, it has **no database layer**: the "repository"
is an HTTP **Adapter** to `ops-core-api` sitting behind the `ChunkRetriever` interface,
so a real DB-backed repository could replace it without touching the service.

```
api/v1  (POST /rag-chat/ask)
  └─ services      RagService: retrieve → confidence gate → generate
       ├─ interfaces/ChunkRetriever  ←impl─ repositories/OpsCoreChunkRetriever (HTTP Adapter)
       └─ interfaces/LLMClient        ←impl─ llm/OpenAICompatibleLLMClient (OpenAI / Ollama)
  core / exceptions / bootstrap (settings, JSON logging, DI container)
```

Patterns used (and commented in code): **Adapter** (HTTP client behind a
repository-style interface), **Repository** (`ChunkRetriever` abstraction),
**Factory** (`create_ops_core_retriever`, `create_llm_client`), **Template Method**
(the two-phase `RagService.ask`), and **Dependency Inversion** (the service depends
only on interfaces).

## Endpoint

`POST /rag-chat/ask`

```jsonc
// request
{ "question": "сколько стоит мезотерапия" }

// response
{
  "answer": "Мезотерапия лица стоит 3500 рублей.",
  "sources": [
    { "document_title": "Прайс-лист", "chunk_index": 2, "score": 0.82 }
  ]
}
```

Health probes: `GET /health/live`, `GET /health/ready` (readiness pings the LLM).

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
# In-topic question → grounded answer with its source
curl -sX POST http://localhost:8001/rag-chat/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "сколько стоит мезотерапия"}'

# Off-topic question → canned fallback, LLM never called (see logs: "llm_skipped": true)
curl -sX POST http://localhost:8001/rag-chat/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "какая погода на марсе"}'
```

## Tests

```bash
uv run pytest --cov=src/app/services --cov-report=term-missing
uv run ruff check .
uv run mypy src
```

Unit tests mock the `ChunkRetriever` and `LLMClient` interfaces and cover both DoD
scenarios (confident answer with source; below-threshold rejection with no LLM call),
plus generate-phase failure handling and language fidelity. Integration tests drive
`POST /rag-chat/ask` through `httpx.ASGITransport` with the service dependency
overridden — no real `ops-core-api` or LLM is contacted.

## Configuration

All configuration is via environment variables (see [`.env.example`](.env.example)):
`OPS_CORE_*` (upstream connection), `RAG_*` (business rules incl. the score
threshold), `LLM_*` (provider/model), `LOG_*`, and `CORS_ALLOW_ORIGINS`. Nothing is
hard-coded; there is no database.

---

# RAG Chat Service (RU)

Тонкий RAG-чатбот **поверх `ops-core-api`**. Сам документы не хранит: на каждый вопрос
он (1) вызывает `POST /api/v1/documents/search` в `ops-core-api` за релевантными
чанками, затем (2) просит LLM сформировать ответ **строго на основе полученного
контекста**.

Ключевое бизнес-правило гейтит шаг 2: если `score` лучшего чанка ниже порога
(`RAG_SCORE_THRESHOLD`, по умолчанию `0.5`) — сервис возвращает заранее заданный ответ
*«не могу ответить точно, уточните у администратора»* и **не обращается к LLM вообще**
(экономия и честность — не даём модели домысливать при слабом retrieval).

## Архитектура

Слоистая архитектура со строгим правилом однонаправленных зависимостей (внешние слои
зависят от внутренних, не наоборот) — см. [`docs/architecture.md`](docs/architecture.md).
Так как сервис ничего не хранит, у него **нет слоя БД**: «репозиторий» — это HTTP
**Adapter** к `ops-core-api` за интерфейсом `ChunkRetriever`, поэтому прямой
DB-репозиторий можно подставить, не трогая сервис.

Применённые паттерны (прокомментированы в коде): **Adapter**, **Repository**,
**Factory**, **Template Method** (двухфазный `RagService.ask`) и **Dependency
Inversion** (сервис зависит только от интерфейсов).

## Эндпоинт

`POST /rag-chat/ask` — принимает `question`, возвращает `answer` и список `sources`
(`document_title` + `chunk_index` + `score`).

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
uv run pytest --cov=src/app/services --cov-report=term-missing
uv run ruff check . && uv run mypy src
```

Юнит-тесты мокают интерфейсы `ChunkRetriever` и `LLMClient` и покрывают оба сценария
DoD (точный ответ с источником; отклонение по порогу без вызова LLM), а также обработку
сбоя генерации и языковую вер­ность. Интеграционные тесты дёргают `POST /rag-chat/ask`
через `httpx.ASGITransport` с подменённым сервисом — реальные `ops-core-api` и LLM не
вызываются.
