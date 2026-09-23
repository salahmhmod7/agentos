"""Quick test: does the agent produce citations?"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.llm import LLM, build_provider
from app.agent.loop import run_agent
from app.agent.rag.store import init_store
from app.db.base import init_db
from app.eval.metrics import has_citation


def main() -> None:
    init_db()
    init_store()

    llm = LLM(provider=build_provider("ollama"))

    question = "According to Python for Data Analysis, what does dropna() do by default?"
    print(f"QUESTION: {question}\n")
    print("Running (may take 1-3 minutes on Ollama)...\n")

    state, _ = run_agent(user_message=question, llm=llm)

    print("=" * 70)
    print("FINAL ANSWER:")
    print("=" * 70)
    print(state.final_answer)
    print("=" * 70)
    print(f"has_citation: {has_citation(state.final_answer or '')}")
    print(f"iterations: {state.iteration}")
    print(f"tools used: {[tc.name for tc in state.tool_calls]}")


if __name__ == "__main__":
    main()
