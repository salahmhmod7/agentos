"""Test the LangGraph agent with checkpointing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.graph.builder import run_graph_agent
from app.agent.graph.checkpointer import get_checkpointer
from app.agent.rag.store import init_store
from app.db.base import init_db


def run_case(prompt: str) -> str:
    print("=" * 70)
    print(f"USER: {prompt}")
    print("-" * 70)

    state, conv_id, thread_id = run_graph_agent(prompt)

    print(f"Conversation: #{conv_id} | Thread: {thread_id}")
    print(f"Iterations: {state.iteration} | Status: {state.status}")

    if state.tool_calls:
        print("Tool calls:")
        for tc in state.tool_calls:
            status = "OK" if tc.error is None else f"ERROR: {tc.error}"
            print(f"  [{tc.iteration}] {tc.name}({tc.arguments}) — {status}")

    print(f"\nFINAL ANSWER:\n{state.final_answer}\n")
    return thread_id


def inspect_thread(thread_id: str) -> None:
    """Show all checkpoints saved for a thread."""
    print("=" * 70)
    print(f"CHECKPOINTS for thread {thread_id}")
    print("-" * 70)

    with get_checkpointer() as cp:
        config = {"configurable": {"thread_id": thread_id}}
        count = 0
        for checkpoint in cp.list(config):
            count += 1
            # Each checkpoint has a `checkpoint` dict with the state snapshot
            cid = checkpoint.config["configurable"].get("checkpoint_id", "?")
            state = checkpoint.checkpoint.get("channel_values", {})
            iteration = state.get("iteration", "?")
            status = state.get("status", "?")
            print(f"  [{count}] checkpoint_id={cid}")
            print(f"      iteration={iteration} | status={status}")
        print(f"\n  Total checkpoints: {count}\n")


if __name__ == "__main__":
    init_db()
    init_store()

    # Case 1: calculator
    tid1 = run_case("What is 3482 * 9217?")

    # Case 2: knowledge search
    tid2 = run_case("According to Python for Data Analysis, what does dropna() do by default?")

    # Inspect checkpoints for the first run
    inspect_thread(tid1)