from fastapi import APIRouter

# Top-level /api/v1 aggregator. Feature routers with the /api/v1 prefix are
# included here; the rag-chat router is mounted separately (see main.py) because
# the task pins its path to /rag-chat/ask (no /api/v1 prefix).
api_router = APIRouter(prefix="/api/v1")
