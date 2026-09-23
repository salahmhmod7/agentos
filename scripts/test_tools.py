"""Test the tool registry and calculator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.tools.registry import build_default_registry


def main() -> None:
    registry = build_default_registry()

    print(f"Registered tools: {registry.names()}")
    print("-" * 50)

    # Test 1: simple multiplication
    result = registry.execute("calculator", {"expression": "25 * 37"})
    print(f"25 * 37 = {result}")
    assert result == 925, f"Expected 925, got {result}"

    # Test 2: more complex
    result = registry.execute("calculator", {"expression": "(12 + 8) / 4"})
    print(f"(12 + 8) / 4 = {result}")
    assert result == 5.0

    # Test 3: power
    result = registry.execute("calculator", {"expression": "2 ** 10"})
    print(f"2 ** 10 = {result}")
    assert result == 1024

    # Test 4: unsafe input should fail
    try:
        registry.execute("calculator", {"expression": "__import__('os').system('ls')"})
        print("❌ SECURITY FAILURE: malicious code was executed!")
    except (ValueError, SyntaxError) as e:
        print(f"✅ Blocked unsafe input: {type(e).__name__}")

    print("-" * 50)
    print("All tests passed!")


if __name__ == "__main__":
    main()