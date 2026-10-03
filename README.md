# RAG Chat Service

[![CI](https://github.com/upkero/rag-chat-service/actions/workflows/ci.yml/badge.svg)](https://github.com/upkero/rag-chat-service/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

A thin Retrieval-Augmented-Generation chatbot built **on top of `ops-core-api`**. It
stores no documents of its own: on each question it (1) calls `ops-core-api`'s
`POST /api/v1/documents/search` to retrieve the most relevant document chunks, then
(2) asks an LLM to answer **strictly from that retrieved context**.

A hard business rule gates step 2: if the best chunk's `score` is below the active
threshold, the service returns a canned *"can't answer precisely, ask an administrator"*
message **without calling the LLM at all** — cost control plus honesty (no hallucinating
on weak retrieval). Which threshold is active depends on whether the optional
[reranker](#reranking-optional) is on:

| Reranker | Gate on the best chunk | `score` is |
|---|---|---|
| off (default) | `RAG_SCORE_THRESHOLD`, default `0.5` | cosine similarity from `ops-core-api` |
| on | `RERANK_MIN_SCORE`, default `0.08` | the cross-encoder's relevance score |

The gate is not the only refusal. A question that clears it can still have nothing
relevant in the context (small talk scores just above the rerank gate); the model then
answers with a `NO_ANSWER` marker, which the service swaps for the same canned message
with empty `sources`. That path **does** call the LLM. The canned message is Russian when
the question contains Cyrillic and English otherwise.

## Architecture

Layered architecture with a strict inward dependency rule (outer layers depend on
inner, never the reverse). Because this service persists nothing, it has **no database
layer** and therefore no `repositories/`: what would be one is a `gateways/` **Adapter**
onto `ops-core-api` behind the `KnowledgeGateway` port. A `repository` is a database
you own; a `gateway` is somebody else's service across a network — it can throttle you
or be down while your own process is perfectly healthy.

```
api/v1  (POST /api/v1/ask)
  └─ services    RagService: retrieve → (rerank) → confidence gate → generate
       ├─ interfaces/KnowledgeGateway ←impl─ gateways/OpsCoreKnowledgeGateway (HTTP Adapter)
       ├─ interfaces/Reranker         ←impl─ gateways/HttpReranker | LocalReranker (optional)
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
file. The rules are the `system` message; the retrieved excerpts and the question go in a
delimited `user` message and are declared to be data, not instructions. Text a user reads
verbatim is not a prompt and lives in `src/app/messages/`.

## Endpoint

`POST /api/v1/ask` — **requires** an `X-API-Key` header.

```jsonc
// request
{ "question": "how do I cancel an appointment" }

// response (reranker on; with it off `score` is the cosine similarity, e.g. 0.82)
{
  "answer": "You can cancel an appointment free of charge up to twenty-four hours before the scheduled start time.",
  "sources": [
    { "document_title": "Cancellation and rescheduling policy", "chunk_index": 0, "score": 0.674 }
  ]
}
```

`sources` lists only the chunks the model was given: those at or above the active
threshold **and** at least half the best score. A weaker neighbour is neither cited nor
sent to the model. An empty `sources` means a refusal.

A blank question is a `422`; an exhausted own rate limit is a `429` with `Retry-After`.

The examples throughout this README quote `ops-core-api`'s seeded demo documents, so
they reproduce against a backend seeded with `uv run python -m src.app.cli.seed`.

Health probes (open, no key): `GET /health/live`, `GET /health/ready` (readiness
checks **both** the LLM and ops-core-api, returns 503 if either is down).

## Reranking (optional)

Embedding search compares two vectors that never saw each other. With
`RERANK_ENABLED=true` the service retrieves `RERANK_CANDIDATES` chunks (default 10),
lets a cross-encoder read the question and each chunk together, keeps the best
`RAG_MAX_CONTEXT_CHUNKS`, and gates on `RERANK_MIN_SCORE` instead of
`RAG_SCORE_THRESHOLD`.

- **Providers.** `RERANK_PROVIDER=http` (default) calls any Cohere-style
  `POST {RERANK_BASE_URL}/rerank` — OpenRouter by default, model `cohere/rerank-v3.5`,
  key in `RERANK_API_KEY`. `RERANK_PROVIDER=local` runs a sentence-transformers
  CrossEncoder (`RERANK_LOCAL_MODEL`, default `BAAI/bge-reranker-v2-m3`, about 2 GB of
  weights) in this process; it needs the image built with `LOCAL_MODELS=true`
  (CPU torch, about 1 GB more) and the model loads at boot.
- **Scale.** Rerank scores are not cosine similarities. On Cohere rerank-v3.5 relevant
  chunks scored 0.10–0.70 and clearly off-topic ones 0.04 or less; small talk lands
  around 0.05–0.10, which is why the `NO_ANSWER` path above exists. The `0.08` default
  is calibrated on Cohere: another reranker (the local model's sigmoid scores included)
  needs its own `RERANK_MIN_SCORE`.
- **Data flow.** With the `http` provider every question and the text of every candidate
  chunk is sent to `RERANK_BASE_URL` — OpenRouter, and from there Cohere. Leave
  reranking off (the default) if document text must not leave your infrastructure
  beyond the LLM call. The `local` provider sends nothing anywhere.
- **Failure.** A reranker that errors or answers with a malformed response is never
  fatal: the service logs a warning and falls back to the embedding order and
  `RAG_SCORE_THRESHOLD`.

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
- Secrets only via env (`SECURITY_API_KEY`, `OPS_CORE_API_KEY`, `LLM_API_KEY`,
  `RERANK_API_KEY`), held as `SecretStr` so they stay out of reprs and errors — none
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
the score threshold), `RERANK_*` (optional reranker: `ENABLED`, `PROVIDER`, `BASE_URL`,
`API_KEY`, `MODEL`, `LOCAL_MODEL`, `CANDIDATES`, `TIMEOUT_SECONDS`, `MIN_SCORE`),
`LOCAL_MODELS` (image build arg that bakes in sentence-transformers for
`RERANK_PROVIDER=local`), `LLM_*` (provider/model), `LOG_*`, and `CORS_ALLOWED_ORIGINS`.
Nothing is hard-coded; there is no database.

## License

[MIT](LICENSE) © 2026 upkero

---

# RAG Chat Service (RU)

Тонкий RAG-чатбот **поверх `ops-core-api`**. Сам документы не хранит: на каждый вопрос
он (1) вызывает `POST /api/v1/documents/search` в `ops-core-api` за релевантными
чанками, затем (2) просит LLM сформировать ответ **строго на основе полученного
контекста**.

Ключевое бизнес-правило гейтит шаг 2: если `score` лучшего чанка ниже действующего
порога — сервис возвращает заранее заданный ответ *«не могу ответить точно, уточните у
администратора»* (см. `src/app/messages/`) и **не обращается к LLM вообще** (экономия и
честность — не даём модели домысливать при слабом retrieval). Какой порог действует,
зависит от [реранкера](#реранкер-опционально): без него — `RAG_SCORE_THRESHOLD` (по
умолчанию `0.5`, `score` — косинусная близость из `ops-core-api`), с ним —
`RERANK_MIN_SCORE` (по умолчанию `0.08`, `score` — оценка кросс-энкодера).

Это не единственный отказ. Вопрос, прошедший гейт, всё равно может не иметь ответа в
контексте (small talk едва проходит порог реранкера): тогда модель отвечает маркером
`NO_ANSWER`, а сервис подменяет его тем же готовым ответом с пустыми `sources`. Этот путь
**вызывает** LLM. Готовый ответ русский, если в вопросе есть кириллица, иначе английский.

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
задаётся плейсхолдером `{reply_language}`, а не переводом файла. Правила — это
`system`-сообщение, а найденные фрагменты и вопрос идут в `user`-сообщении между
разделителями и объявлены данными, а не инструкциями. Текст, который пользователь читает
дословно, — не промпт и лежит в `src/app/messages/`.

Схема слоёв: `RagService` — retrieve → (rerank) → гейт → generate; за портами стоят
`KnowledgeGateway` (`OpsCoreKnowledgeGateway`), `Reranker` (`HttpReranker` или
`LocalReranker`, опционально) и `LLMClient`.

## Эндпоинт

`POST /api/v1/ask` — принимает `question`, возвращает `answer` и список `sources`
(`document_title` + `chunk_index` + `score`). **Требует заголовок `X-API-Key`.**

В `sources` попадают только фрагменты, которые получила модель: не ниже действующего
порога **и** не ниже половины лучшего `score`. Пустой `sources` — это отказ. Пустой
вопрос — `422`; собственный лимит запросов — `429` с `Retry-After`.

## Реранкер (опционально)

С `RERANK_ENABLED=true` сервис берёт `RERANK_CANDIDATES` чанков (по умолчанию 10),
кросс-энкодер читает вопрос и каждый чанк вместе, остаются лучшие
`RAG_MAX_CONTEXT_CHUNKS`, а гейт работает по `RERANK_MIN_SCORE` вместо
`RAG_SCORE_THRESHOLD`.

- **Провайдеры.** `RERANK_PROVIDER=http` (по умолчанию) вызывает любой Cohere-совместимый
  `POST {RERANK_BASE_URL}/rerank` — по умолчанию OpenRouter, модель
  `cohere/rerank-v3.5`, ключ в `RERANK_API_KEY`. `RERANK_PROVIDER=local` запускает
  sentence-transformers CrossEncoder (`RERANK_LOCAL_MODEL`, по умолчанию
  `BAAI/bge-reranker-v2-m3`, около 2 ГБ весов) внутри процесса; нужен образ, собранный
  с `LOCAL_MODELS=true`, модель грузится при старте.
- **Шкала.** Оценки реранкера — не косинусная близость. На Cohere rerank-v3.5
  релевантные чанки получали 0.10–0.70, явный оффтоп — 0.04 и ниже, small talk —
  около 0.05–0.10 (его ловит путь `NO_ANSWER`). Порог `0.08` откалиброван под Cohere:
  другому реранкеру (в том числе локальному) нужен свой `RERANK_MIN_SCORE`.
- **Куда уходят данные.** С провайдером `http` каждый вопрос и текст каждого
  чанка-кандидата отправляется на `RERANK_BASE_URL` — в OpenRouter, а оттуда в Cohere.
  Если текст документов не должен покидать вашу инфраструктуру, не включайте реранкер
  (по умолчанию он выключен). `local` ничего никуда не отправляет.
- **Сбой.** Ошибка или кривой ответ реранкера не фатальны: сервис пишет warning и
  возвращается к порядку по эмбеддингам и `RAG_SCORE_THRESHOLD`.

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
- Ключи (`SECURITY_API_KEY`, `OPS_CORE_API_KEY`, `LLM_API_KEY`, `RERANK_API_KEY`)
  хранятся как `SecretStr` и не попадают в repr и тексты ошибок.

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
