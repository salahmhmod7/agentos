"""Collect all API routers under /api prefix."""

from fastapi import APIRouter

from app.api.routes.chat import router as chat_router
from app.api.routes.conversations import router as conversations_router
from app.api.routes.documents import router as documents_router
from app.api.routes.eval import router as eval_router
from app.api.routes.graph_chat import router as graph_chat_router
from app.api.routes.health import router as health_router
from app.api.routes.stats import router as stats_router
from app.api.routes.stream import router as stream_router


api_router = APIRouter(prefix="/api")
api_router.include_router(health_router)
api_router.include_router(stats_router)
api_router.include_router(chat_router)
api_router.include_router(stream_router)
api_router.include_router(conversations_router)
api_router.include_router(documents_router)
api_router.include_router(eval_router)
api_router.include_router(graph_chat_router)