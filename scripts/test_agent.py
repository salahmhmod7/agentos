"""Manual test of the agent loop + persistence."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import memory
from app.agent.loop import run_agent
from app.db.base import init_db


def run_case(prompt: str, conversation_id: int | None = None) -> int:
    print("=" * 70)
    print(f"USER: {prompt}")
    print("-" * 70)

    state, conv_id = run_agent(prompt, conversation_id=conversation_id)

    print(f"Conversation: #{conv_id} | Iterations: {state.iteration} | Status: {state.status}")

    if state.tool_calls:
        print("Tool calls:")
        for tc in state.tool_calls:
            status = "OK" if tc.error is None else f"ERROR: {tc.error}"
            print(f"  [{tc.iteration}] {tc.name}({tc.arguments})  ({status})")

    print(f"\nFINAL ANSWER:\n{state.final_answer}\n")
    return conv_id


if __name__ == "__main__":
    # 1. Create tables (idempotent)
    init_db()

    # 2. First turn — creates a new conversation
    conv = run_case("Who was Ada Lovelace?")

    # 3. Second turn — same conversation, agent should remember context
    run_case("When was she born?", conversation_id=conv)

    # 4. Inspect the saved conversation
    print("=" * 70)
    print(f"HISTORY of conversation #{conv}")
    print("-" * 70)
    history = memory.get_conversation_history(conv)
    for run in history.get("runs", []):
        print(f"  Run #{run['id']} | status={run['status']} | tools={run['tool_calls']}")
        print(f"    Q: {run['user_message']}")
        print(f"    A: {(run['final_answer'] or '')[:120]}...")
