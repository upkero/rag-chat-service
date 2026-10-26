import logging
import re
from collections.abc import Sequence

from opentelemetry import trace

from src.app.contracts.rag.rag_answer import RagAnswer
from src.app.contracts.rag.retrieved_chunk import RetrievedChunk
from src.app.core import metrics
from src.app.core.settings.rag import RagSettings
from src.app.core.settings.rerank import RerankSettings
from src.app.interfaces.knowledge.knowledge_gateway import KnowledgeGateway
from src.app.interfaces.llm.llm_client import LLMClient
from src.app.interfaces.rerank.reranker import Reranker
from src.app.llm.skills.answer_from_context import answer_from_context
from src.app.messages import get_message

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer(__name__)

_CYRILLIC = re.compile("[А-Яа-яЁё]")

# A chunk must score at least this fraction of the best one to reach the model and the
# caller. The gate only judges the best chunk, so without a window the tail of the
# candidate list (0.07 behind a 0.67) rode along as "sources". Half is deliberately
# loose: it drops clear stragglers while keeping genuine runners-up on the same topic.
_RELEVANCE_WINDOW = 0.5


def _relevant(chunks: Sequence[RetrievedChunk], threshold: float) -> list[RetrievedChunk]:
    """The chunks worth showing: at or above the active gate (``RERANK_MIN_SCORE`` with a
    reranker, ``RAG_SCORE_THRESHOLD`` without) and within ``_RELEVANCE_WINDOW`` of the best."""
    floor = max(threshold, _RELEVANCE_WINDOW * max(chunk.score for chunk in chunks))
    return [chunk for chunk in chunks if chunk.score >= floor]


class RagService:
    """Orchestrates the RAG flow: retrieve, optionally rerank, apply the confidence gate, generate.

    The service depends only on the KnowledgeGateway, Reranker and LLMClient *interfaces*
    (Dependency Inversion) — real adapters in production, test doubles in unit tests, no
    code change either way.
    """

    def __init__(
        self,
        *,
        knowledge: KnowledgeGateway,
        llm_client: LLMClient,
        settings: RagSettings,
        reranker: Reranker | None = None,
        rerank_settings: RerankSettings | None = None,
    ) -> None:
        self._knowledge = knowledge
        self._llm = llm_client
        self._settings = settings
        self._reranker = reranker
        self._rerank_settings = rerank_settings

    async def ask(self, question: str) -> RagAnswer:
        with _tracer.start_as_current_span("rag.ask") as span:
            # Length and script only: the question's text is never put on a span, a metric or a log.
            span.set_attributes({"question.length": len(question), "question.lang": self._refusal_language(question)})
            outcome, reranked, in_context = "error", False, 0
            try:
                chunks = await self._retrieve(question)
                span.set_attribute("rag.chunks.retrieved", len(chunks))
                threshold = self._settings.score_threshold
                chunks, reranked = await self._rerank(question, chunks)
                if reranked and self._rerank_settings is not None:
                    # Rerank scores live on their own scale, so they get their own gate.
                    threshold = self._rerank_settings.min_score

                # Business rule / confidence gate: with no chunk scoring above the
                # threshold we refuse to call the LLM at all — cheaper, and honest (no
                # hallucinating on weak retrieval).
                best_score = max((chunk.score for chunk in chunks), default=0.0)
                span.set_attributes(
                    {"rag.reranked": reranked, "rag.best_score": best_score, "rag.threshold": threshold}
                )
                metrics.retrieval_chunks.record(len(chunks), {"stage": "after_rerank"})
                if chunks:
                    metrics.top_score.record(best_score, {"scale": "rerank" if reranked else "cosine"})
                if not chunks or best_score < threshold:
                    logger.info(
                        "rag.gate.low_confidence",
                        extra={
                            "chunks_found": len(chunks),
                            "best_score": best_score,
                            "threshold": threshold,
                            "llm_skipped": True,
                        },
                    )
                    outcome = "low_confidence_gate"
                    return RagAnswer(
                        answer=get_message("low_confidence", self._refusal_language(question)),
                        sources=(),
                    )

                chunks = _relevant(chunks, threshold)
                in_context = min(len(chunks), self._settings.max_context_chunks)
                answer = await self._generate(question, chunks)
                # The chunks passed the gate but the model found nothing in them: that is a
                # refusal like any other, so no answer text from the model and no sources.
                outcome = "answered" if answer else "model_no_answer"
                return answer or RagAnswer(
                    answer=get_message("low_confidence", self._refusal_language(question)), sources=()
                )
            finally:
                span.set_attributes({"rag.outcome": outcome, "rag.chunks.in_context": in_context})
                metrics.answers.add(1, {"outcome": outcome, "reranked": reranked})
                if outcome != "error":
                    metrics.retrieval_chunks.record(in_context, {"stage": "in_context"})

    @staticmethod
    def _refusal_language(question: str) -> str:
        """The canned refusal follows the question's script: any Cyrillic gets Russian
        (mixed RU/EN included), everything else English. A Russian question typed in
        Latin letters reads as English here, the same as the model's own guess."""
        return "ru" if _CYRILLIC.search(question) else "en"

    async def _retrieve(self, question: str) -> Sequence[RetrievedChunk]:
        """Phase 1 — pull candidate chunks from the gateway (logged separately)."""
        top_k = self._settings.top_k
        if self._reranker is not None and self._rerank_settings is not None:
            top_k = max(top_k, self._rerank_settings.candidates)
        with _tracer.start_as_current_span("rag.retrieve", attributes={"rag.top_k": top_k}) as span:
            # One logical span: the httpx spans of each attempt (retries included) sit under it.
            chunks = await self._knowledge.search(question, top_k)
            span.set_attribute("rag.chunks.retrieved", len(chunks))
        metrics.retrieval_chunks.record(len(chunks), {"stage": "retrieved"})
        logger.info(
            "rag.retrieve",
            extra={
                "chunks_found": len(chunks),
                "top_score": max((chunk.score for chunk in chunks), default=0.0),
                "scores": [round(chunk.score, 4) for chunk in chunks],
            },
        )
        return chunks

    async def _rerank(
        self, question: str, chunks: Sequence[RetrievedChunk]
    ) -> tuple[Sequence[RetrievedChunk], bool]:
        """Re-score the candidates with the cross-encoder. Never fatal: if the
        reranker is down, fall back to the embedding order and the embedding gate."""
        if self._reranker is None or self._rerank_settings is None or not chunks:
            return chunks[: self._settings.top_k], False
        with _tracer.start_as_current_span("rag.rerank") as span:
            try:
                reranked = await self._reranker.rerank(question, chunks, self._settings.max_context_chunks)
            except Exception as exc:
                # The one silent fallback in this service, so it is counted and flagged.
                span.set_attributes({"rerank.fallback": True, "error.type": type(exc).__name__})
                metrics.rerank_fallbacks.add(1, {"reranker": type(self._reranker).__name__})
                logger.warning("rag.rerank.failed; using embedding order", exc_info=True)
                return chunks[: self._settings.top_k], False
            span.set_attribute("rerank.fallback", False)
        logger.info("rag.rerank", extra={"scores": [round(chunk.score, 4) for chunk in reranked]})
        return reranked, True

    async def _generate(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
    ) -> RagAnswer | None:
        """Phase 2 — hand the grounded question to the skill that answers it.

        Everything model-facing (the prompt, its placeholders, parsing the reply)
        lives in the skill; this service only decides *whether* to ask.
        """
        return await answer_from_context(
            client=self._llm,
            question=question,
            chunks=list(chunks)[: self._settings.max_context_chunks],
        )
