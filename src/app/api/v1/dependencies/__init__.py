from typing import Annotated

from fastapi import Depends, Request

from src.app.bootstrap.container import ApplicationContainer
from src.app.interfaces.knowledge.knowledge_gateway import KnowledgeGateway
from src.app.interfaces.llm.llm_client import LLMClient
from src.app.services.rag import RagService


def get_container(request: Request) -> ApplicationContainer:
    return request.app.state.container  # type: ignore[no-any-return]


ContainerDep = Annotated[ApplicationContainer, Depends(get_container)]


def get_llm_client(container: ContainerDep) -> LLMClient:
    return container.llm_client


def get_knowledge_gateway(container: ContainerDep) -> KnowledgeGateway:
    return container.knowledge_gateway


def get_rag_service(container: ContainerDep) -> RagService:
    return container.rag_service


# Depends helpers exposed as annotated aliases (used directly as param types in
# routers, e.g. `service: RagServiceDep`).
LLMClientDep = Annotated[LLMClient, Depends(get_llm_client)]
KnowledgeGatewayDep = Annotated[KnowledgeGateway, Depends(get_knowledge_gateway)]
RagServiceDep = Annotated[RagService, Depends(get_rag_service)]
