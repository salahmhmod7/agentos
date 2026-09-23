"""Demonstrate Human-in-the-Loop with LangGraph interrupts.

Flow:
  1. Ask the agent to save a report.
  2. The graph pauses before executing `save_report` (requires approval).
  3. We print the pending request.
  4. We resume with Command(resume={"approved": True/False}).
  5. The graph continues and produces the final answer.
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.types import Command

from app.agent.graph.builder import build_graph
from app.agent.graph.checkpointer import get_checkpointer
from app.agent.llm import LLM, build_provider
from app.agent.observability import Tracer, silent_handler
from app.agent.rag.store import init_store
from app.agent.tools.registry import build_default_registry
from app.core.config import settings
from app.db.base import init_db


def main() -> None:
    init_db()
    init_store()

    registry = build_default_registry()
    llm = LLM(provider=build_provider(settings.llm_provider))
    tracer = Tracer([silent_handler])

    with get_checkpointer() as cp:
        graph = build_graph(registry, llm, tracer, checkpointer=cp)

        # Unique thread id per run (so old checkpoints don't pile up)
        thread_id = f"hitl-demo-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}

        user_msg = (
            "Save a short markdown report about Ada Lovelace to a file. "
            "The report should be one paragraph long."
        )

        print("=" * 70)
        print(f"THREAD: {thread_id}")
        print(f"USER: {user_msg}")
        print("=" * 70)

        from app.agent.loop import SYSTEM_PROMPT

        initial_state = {
            "user_message": user_msg,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "iteration": 0,
            "max_iterations": 8,
            "recovery_count": 0,
            "tool_calls": [],
            "needs_tools": False,
            "final_answer": "",
            "status": "running",
        }

        # First invocation — will pause at the interrupt
        result = graph.invoke(initial_state, config=config)

        interrupts = result.get("__interrupt__")
        if not interrupts:
            print("\nNo interrupt — graph finished on its own.")
            print(f"\nFINAL:\n{result.get('final_answer')}")
            return

        print("\n⏸️  GRAPH PAUSED — waiting for human approval")
        print("-" * 70)

        for intr in interrupts:
            payload = intr.value if hasattr(intr, "value") else intr
            print(f"  Action  : {payload.get('action')}")
            print(f"  Tool    : {payload.get('tool')}")
            print(f"  Args    :")
            args = payload.get("arguments", {})
            for k, v in args.items():
                v_str = str(v)
                if len(v_str) > 200:
                    v_str = v_str[:200] + "..."
                print(f"    {k} = {v_str}")

        print()
        answer = input("Approve? [y/N]: ").strip().lower()
        approved = answer in ("y", "yes")

        print(f"\nHuman decision: {'APPROVED' if approved else 'REJECTED'}")
        print("=" * 70)

        # Resume with the decision
        result = graph.invoke(
            Command(resume={"approved": approved}),
            config=config,
        )

        print(f"\nStatus: {result.get('status')}")
        print(f"Iterations: {result.get('iteration')}")

        print(f"\nTool activity:")
        for rec in result.get("tool_calls", []):
            status = "OK" if not rec.get("error") else f"ERR: {rec['error']}"
            print(f"  [{rec['iteration']}] {rec['name']} — {status}")
            if rec["name"] == "save_report" and not rec.get("error") and rec.get("result"):
                print(f"      → saved to {rec['result'].get('path')}")

        print(f"\nFINAL ANSWER:\n{result.get('final_answer')}")


if __name__ == "__main__":
    main()
