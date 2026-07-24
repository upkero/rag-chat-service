from fastapi import APIRouter, HTTPException

from src.app.api.v1.dependencies import LLMClientDep

router = APIRouter(prefix="/health", tags=["infra"])


@router.get("/live")
async def liveness() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def readiness(llm_client: LLMClientDep) -> dict[str, str]:
    if not await llm_client.ping():
        raise HTTPException(status_code=503, detail="LLM unavailable")
    return {"status": "ok"}
