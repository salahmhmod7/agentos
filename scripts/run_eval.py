"""CLI: run the evaluation dataset."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.base import init_db
from app.agent.rag.store import init_store as init_rag_store
from app.eval.runner import run_eval


def main() -> None:
    init_db()
    init_rag_store()
    run_eval(quiet=True, verbose=True)


if __name__ == "__main__":
    main()