"""Metric instruments, defined once.

Names and labels follow OBSERVABILITY-AUDIT §1.2. A label is a bounded set (an
outcome, a stage, a model), never a question, an id or an exception message.
The instruments are created at import against the global meter, which binds to
the real provider as soon as `core.telemetry.setup_metrics` installs it.
"""

from opentelemetry import metrics

_meter = metrics.get_meter("rag-chat-service")

answers = _meter.create_counter(
    "rag.answers", description="Questions answered, by how the answer came about."
)
retrieval_chunks = _meter.create_histogram(
    "rag.retrieval.chunks",
    description="Chunks left at each stage of retrieval.",
    explicit_bucket_boundaries_advisory=[0, 1, 2, 3, 5, 10, 20, 50],
)
top_score = _meter.create_histogram(
    "rag.top_score",
    description="Best chunk score, per scoring scale, for tuning the confidence gate.",
    explicit_bucket_boundaries_advisory=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
rerank_fallbacks = _meter.create_counter(
    "rag.rerank.fallbacks", description="Reranks that failed and fell back to the embedding order."
)
operation_duration = _meter.create_histogram(
    "gen_ai.client.operation.duration",
    unit="s",
    description="LLM request duration.",
    explicit_bucket_boundaries_advisory=[
        0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
    ],
)
token_usage = _meter.create_histogram(
    "gen_ai.client.token.usage",
    unit="{token}",
    description="Tokens per LLM request.",
    explicit_bucket_boundaries_advisory=[1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144],
)
llm_cost = _meter.create_counter(
    "llm.cost", description="LLM spend in USD, as reported by the provider (OpenRouter usage.cost)."
)
finish_reasons = _meter.create_counter(
    "llm.finish_reasons", description="LLM replies by finish reason; `length` means truncation."
)
