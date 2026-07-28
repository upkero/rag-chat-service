import logging
from collections.abc import Sequence

from src.app.contracts.rag.rag_answer import RagAnswer
from src.app.contracts.rag.retrieved_chunk import RetrievedChunk
from src.app.core.settings.rag import RagSettings
from src.app.interfaces.knowledge.knowledge_gateway import KnowledgeGateway
from src.app.interfaces.llm.llm_client import LLMClient
from src.app.llm.skills.answer_from_context import answer_from_context
from src.app.messages import get_message

logger = logging.getLogger(__name__)


class RagService:
    """Orchestrates the two-phase RAG flow: retrieve → (confidence gate) → generate.

    Template Method (lightweight): ``ask()`` fixes the skeleton and delegates to
    the ``_retrieve`` and ``_generate`` steps. The service depends only on the
    KnowledgeGateway and LLMClient *interfaces* (Dependency Inversion) — real
    adapters in production, test doubles in unit tests, no code change either way.
    """

    def __init__(
        self,
        *,
        knowledge: KnowledgeGateway,
        llm_client: LLMClient,
        settings: RagSettings,
    ) -> None:
        self._knowledge = knowledge
        self._llm = llm_client
        self._settings = settings

    async def ask(self, question: str) -> RagAnswer:
        chunks = await self._retrieve(question)

        # Business rule / confidence gate: with no chunk scoring above the
        # threshold we refuse to call the LLM at all — cheaper, and honest (no
        # hallucinating on weak retrieval).
        best_score = max((chunk.score for chunk in chunks), default=0.0)
        if not chunks or best_score < self._settings.score_threshold:
            logger.info(
                "rag.gate.low_confidence",
                extra={
                    "chunks_found": len(chunks),
                    "best_score": best_score,
                    "threshold": self._settings.score_threshold,
                    "llm_skipped": True,
                },
            )
            return RagAnswer(
                answer=get_message("low_confidence", self._settings.message_language),
                sources=(),
                from_llm=False,
            )

        return await self._generate(question, chunks)

    async def _retrieve(self, question: str) -> Sequence[RetrievedChunk]:
        """Phase 1 — pull candidate chunks from the gateway (logged separately)."""
        chunks = await self._knowledge.search(question, self._settings.top_k)
        logger.info(
            "rag.retrieve",
            extra={
                "chunks_found": len(chunks),
                "top_score": max((chunk.score for chunk in chunks), default=0.0),
                "scores": [round(chunk.score, 4) for chunk in chunks],
            },
        )
        return chunks

    async def _generate(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
    ) -> RagAnswer:
        """Phase 2 — hand the grounded question to the skill that answers it.

        Everything model-facing (the prompt, its placeholders, parsing the reply)
        lives in the skill; this service only decides *whether* to ask.
        """
        return await answer_from_context(
            client=self._llm,
            question=question,
            chunks=list(chunks)[: self._settings.max_context_chunks],
        )
