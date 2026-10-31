from functools import cached_property

from src.app.core.settings.llm import get_llm_settings
from src.app.core.settings.ops_core import get_ops_core_settings
from src.app.core.settings.rag import get_rag_settings
from src.app.core.settings.rerank import get_rerank_settings
from src.app.gateways.factory import create_ops_core_knowledge_gateway, create_reranker
from src.app.interfaces.knowledge.knowledge_gateway import KnowledgeGateway
from src.app.interfaces.llm.llm_client import LLMClient
from src.app.interfaces.rerank.reranker import Reranker
from src.app.llm.factory import create_llm_client
from src.app.services.rag import RagService


class ApplicationContainer:
    """Manual DI container.

    Each dependency is a lazily-built, memoized ``cached_property``; return types
    are interfaces (not concretes), so the wiring reads as a composition of
    abstractions. Built once on first access and closed on lifespan shutdown.
    """

    @cached_property
    def llm_client(self) -> LLMClient:
        return create_llm_client(get_llm_settings())

    @cached_property
    def knowledge_gateway(self) -> KnowledgeGateway:
        return create_ops_core_knowledge_gateway(get_ops_core_settings())

    @cached_property
    def reranker(self) -> Reranker | None:
        return create_reranker(get_rerank_settings())

    @cached_property
    def rag_service(self) -> RagService:
        return RagService(
            knowledge=self.knowledge_gateway,
            llm_client=self.llm_client,
            settings=get_rag_settings(),
            reranker=self.reranker,
            rerank_settings=get_rerank_settings(),
        )

    async def close(self) -> None:
        # Close every dependency that was built and has a close(); rag_service
        # holds the same clients but has no close() of its own, so it is skipped.
        for dependency in tuple(self.__dict__.values()):
            close = getattr(dependency, "close", None)
            if callable(close):
                await close()
