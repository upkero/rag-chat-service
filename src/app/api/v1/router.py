from fastapi import APIRouter, Depends

from src.app.api.v1.dependencies.rate_limit import enforce_rate_limit
from src.app.api.v1.dependencies.security import require_api_key
from src.app.api.v1.routers.rag import router as rag_router

# Top-level /api/v1 aggregator — includes every versioned feature router, so
# their paths all share the /api/v1 prefix (e.g. /api/v1/ask), consistent with
# the other portfolio services.
#
# Group-level dependencies apply to EVERY route below: rate-limit first (cheap,
# by client IP, throttles brute force before auth), then API-key auth. Health
# probes live outside this group and stay open.
api_router = APIRouter(
    prefix="/api/v1",
    dependencies=[Depends(enforce_rate_limit), Depends(require_api_key)],
)
api_router.include_router(rag_router)
