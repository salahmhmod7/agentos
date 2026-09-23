"""Health check endpoint."""

from fastapi import APIRouter

from app.api.schemas import HealthResponse
from app.core.config import settings


router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness + basic config check."""
    db_kind = settings.database_url.split(":")[0] if settings.database_url else "unknown"
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        model=settings.ollama_model,
        database=db_kind,
    )