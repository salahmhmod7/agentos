"""Calculator tool — safe arithmetic evaluation."""

import ast
import operator
from typing import Any

from app.agent.tools.base import Tool


# Only these operators are allowed — no eval, no imports, no functions
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate a restricted AST."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")

    if isinstance(node, ast.BinOp):
        op = _ALLOWED_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_safe_eval(node.left), _safe_eval(node.right))

    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op(_safe_eval(node.operand))

    raise ValueError(f"Unsupported expression: {type(node).__name__}")


class CalculatorTool(Tool):
    """Evaluate simple arithmetic expressions safely."""

    name = "calculator"
    description = (
        "Evaluate a mathematical expression. "
        "Supports +, -, *, /, ** (power), and % (modulo). "
        "Use this for ANY arithmetic instead of computing yourself."
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The math expression, e.g. '25 * 37' or '(12 + 8) / 4'",
            }
        },
        "required": ["expression"],
    }

    def run(self, expression: str, **_: Any) -> float:
        """Safely evaluate the expression and return a number."""
        tree = ast.parse(expression, mode="eval")
        return _safe_eval(tree.body)