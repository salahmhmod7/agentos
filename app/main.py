"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import api_router
from app.core.config import settings
from app.db.base import init_db
from app.agent.rag.store import init_store as init_rag_store


STATIC_DIR = Path(__file__).resolve().parent.parent / "data" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    init_rag_store()
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Autonomous AI Research Agent",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(api_router)


# ----- UI routes -----

@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/chat", include_in_schema=False)
def chat_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "chat.html")


@app.get("/documents", include_in_schema=False)
def documents_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "documents.html")


@app.get("/eval", include_in_schema=False)
def eval_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "eval.html")


@app.get("/graph-chat", include_in_schema=False)
def graph_chat_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "graph_chat.html")