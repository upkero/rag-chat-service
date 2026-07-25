from fastapi import APIRouter, HTTPException

from src.app.api.v1.dependencies import ChunkRetrieverDep, LLMClientDep

router = APIRouter(prefix="/health", tags=["infra"])


@router.get("/live")
async def liveness() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def readiness(
    llm_client: LLMClientDep,
    retriever: ChunkRetrieverDep,
) -> dict[str, str]:
    # Ready only if BOTH downstream dependencies respond: the LLM and the
    # ops-core-api retrieval backend. Either being down makes the service unable
    # to answer, so it should fail readiness (and be pulled from the LB).
    checks = {
        "llm": await llm_client.ping(),
        "ops_core": await retriever.ping(),
    }
    unavailable = [name for name, ok in checks.items() if not ok]
    if unavailable:
        raise HTTPException(
            status_code=503,
            detail=f"Dependencies unavailable: {', '.join(unavailable)}",
        )
    return {"status": "ok"}
