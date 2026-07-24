from fastapi import APIRouter

from src.app.api.v1.dependencies import RagServiceDep
from src.app.schemas.rag import AskRequest, AskResponse

# The task pins this exact path (/rag-chat/ask), so this router is mounted
# directly in main.py rather than under the /api/v1 aggregator.
router = APIRouter(prefix="/rag-chat", tags=["rag"])


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, service: RagServiceDep) -> AskResponse:
    # schema in -> service works in contracts -> schema out.
    answer = await service.ask(body.question)
    return AskResponse.from_contract(answer)
