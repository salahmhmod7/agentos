"""Manual smoke test: verify Ollama + our LLM wrapper work."""

import sys
from pathlib import Path

# Add project root to sys.path so 'app' is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.llm import llm


def main() -> None:
    messages = [
        {
            "role": "system",
            "content": "You are a concise assistant. Reply in one short sentence.",
        },
        {
            "role": "user",
            "content": "What is LangGraph in one sentence?",
        },
    ]

    print(f"Model: {llm.model}")
    print("-" * 50)

    response = llm.chat(messages)
    print(response.content)


if __name__ == "__main__":
    main()