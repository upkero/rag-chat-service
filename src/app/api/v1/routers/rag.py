from fastapi import APIRouter

from src.app.api.v1.dependencies import RagServiceDep
from src.app.schemas.rag import AskRequest, AskResponse

# Included by the /api/v1 aggregator (see api/v1/router.py), so the full path is
# /api/v1/ask. No feature name in the path: the host and port already say which
# service this is, and repeating it there is a stutter.
router = APIRouter(tags=["rag"])


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, service: RagServiceDep) -> AskResponse:
    # schema in -> service works in contracts -> schema out.
    answer = await service.ask(body.question)
    return AskResponse.from_contract(answer)
