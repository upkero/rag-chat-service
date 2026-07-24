from fastapi import APIRouter

from src.app.api.v1.routers.rag import router as rag_router

# Top-level /api/v1 aggregator — includes every versioned feature router, so
# their paths all share the /api/v1 prefix (e.g. /api/v1/rag-chat/ask),
# consistent with the other portfolio services.
api_router = APIRouter(prefix="/api/v1")
api_router.include_router(rag_router)
