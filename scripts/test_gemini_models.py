"""Try several Gemini models to find one that works for tool calling."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from app.core.config import settings
from app.agent.tools.registry import build_default_registry


# Priority order: lite versions have the most generous free tier
CANDIDATES = [
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
]


def try_model(model: str, tools: list[dict]) -> str:
    """Try one model. Returns 'ok', 'quota', 'notfound', 'error:<msg>'."""
    try:
        client = OpenAI(
            api_key=settings.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "what is 5*3?"}],
            tools=tools,
        )
        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None) or []
        if calls:
            return f"ok (called: {[c.function.name for c in calls]})"
        return f"ok (no tools, content: {(msg.content or '')[:40]})"
    except Exception as e:
        s = str(e)
        if "429" in s or "quota" in s.lower():
            return "quota_exceeded"
        if "404" in s or "not available" in s or "NOT_FOUND" in s:
            return "not_available"
        return f"error: {s[:120]}"


def main() -> None:
    registry = build_default_registry()
    tools = registry.schemas()

    print("=" * 70)
    print("TESTING GEMINI MODELS (tool calling)")
    print("=" * 70)

    for model in CANDIDATES:
        print(f"\n>>> {model}")
        result = try_model(model, tools)
        print(f"    → {result}")


if __name__ == "__main__":
    main()