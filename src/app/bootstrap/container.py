import inspect
from functools import cached_property

from src.app.core.settings.llm import get_llm_settings
from src.app.core.settings.ops_core import get_ops_core_settings
from src.app.core.settings.rag import get_rag_settings
from src.app.gateways.factory import create_ops_core_knowledge_gateway
from src.app.interfaces.knowledge.knowledge_gateway import KnowledgeGateway
from src.app.interfaces.llm.llm_client import LLMClient
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
    def rag_service(self) -> RagService:
        return RagService(
            knowledge=self.knowledge_gateway,
            llm_client=self.llm_client,
            settings=get_rag_settings(),
        )

    async def close(self) -> None:
        # Generic teardown: close every instantiated dependency exposing close(),
        # de-duplicated by identity (rag_service shares the same client instances,
        # so each underlying client is closed exactly once).
        closed_dependency_ids: set[int] = set()
        for dependency in tuple(self.__dict__.values()):
            dependency_id = id(dependency)
            if dependency_id in closed_dependency_ids:
                continue
            close = getattr(dependency, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
            closed_dependency_ids.add(dependency_id)
