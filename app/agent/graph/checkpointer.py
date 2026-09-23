"""Persistent checkpoint storage for the LangGraph agent.

Uses SqliteSaver so the graph state survives server restarts and can
be resumed from any point (used later for Human-in-the-loop).
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from langgraph.checkpoint.sqlite import SqliteSaver

from app.core.config import settings


def _resolve_db_path() -> str:
    url = settings.database_url
    if url.startswith("sqlite:///"):
        return url.replace("sqlite:///", "", 1)
    return url


DB_PATH = _resolve_db_path()


@contextmanager
def get_checkpointer() -> Iterator[SqliteSaver]:
    """Open a SqliteSaver bound to agentos.db.

    Yields a saver and closes the underlying connection on exit.
    """
    saver = SqliteSaver.from_conn_string(DB_PATH)
    with saver as s:
        yield s