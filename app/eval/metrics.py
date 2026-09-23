"""Metric functions for evaluating agent runs."""

from typing import Any

from app.agent.state import AgentState


NOT_FOUND_PHRASES = [
    "not found",
    "not in the book",
    "not supported",
    "not covered",
    "does not mention",
    "doesn't mention",
    "no mention",
    "cannot find",
    "could not find",
    "not appear",
    "no information",
    "not explicitly",
]

CITATION_MARKERS = [
    "【",              # 【source: ...】
    "source:",
    "chunk",
    "chunk_index",
    "according to",
    ".pdf",
    ".txt",
    ".md",
    "[handbook",       # [handbook.pdf, chunk 0]
    "in the book",
    "the book states",
    "the book shows",
]


def used_tool_names(state: AgentState) -> list[str]:
    return [tc.name for tc in state.tool_calls]


def is_tool_correct(state: AgentState, expected_tool: str) -> bool:
    """Did the agent call the expected tool at least once?"""
    return expected_tool in used_tool_names(state)


def is_retrieval_hit(state: AgentState) -> bool | None:
    """For knowledge_search: did it return non-empty results?"""
    ks_calls = [tc for tc in state.tool_calls if tc.name == "knowledge_search"]
    if not ks_calls:
        return None
    for tc in ks_calls:
        if isinstance(tc.result, dict) and tc.result.get("count", 0) > 0:
            return True
    return False


def count_keywords(answer: str, keywords: list[str]) -> tuple[int, int]:
    """Return (found, total)."""
    if not keywords:
        return 0, 0
    answer_lc = (answer or "").lower()
    found = sum(1 for k in keywords if k.lower() in answer_lc)
    return found, len(keywords)


def is_boundary_compliant(answer: str) -> bool:
    """Did the answer say 'not found' or similar?"""
    a = (answer or "").lower()
    return any(p in a for p in NOT_FOUND_PHRASES)


def has_citation(answer: str) -> bool:
    """Does the answer contain any citation markers?"""
    a = answer or ""
    return any(m in a for m in CITATION_MARKERS)


def detect_hallucination(state: AgentState, expected_answer_type: str) -> bool | None:
    """Heuristic: hallucinates if the item expects 'not_found' but answer claims presence.

    Returns None (N/A) if the run itself failed — a failed run isn't a hallucination.
    """
    if expected_answer_type != "not_found":
        return None
    if state.status != "done":
        return None  # N/A — the run didn't complete
    return not is_boundary_compliant(state.final_answer or "")

def evaluate_run(item: dict, state: AgentState, latency_s: float) -> dict[str, Any]:
    """Evaluate one run against one dataset item. Returns a metrics dict."""
    answer = state.final_answer or ""
    expected_tool = item.get("expected_tool")
    expected_type = item.get("expected_answer_type", "present")
    keywords = item.get("expected_keywords", []) or []

    kw_found, kw_total = count_keywords(answer, keywords)

    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],

        # Run status
        "status": state.status,
        "iterations": state.iteration,
        "latency_s": round(latency_s, 2),
        "failure": state.status not in ("done",),

        # Tool usage
        "tools_used": used_tool_names(state),
        "tool_correct": is_tool_correct(state, expected_tool) if expected_tool else None,

        # Retrieval
        "retrieval_hit": is_retrieval_hit(state) if expected_tool == "knowledge_search" else None,

        # Answer quality
        "keywords_found": kw_found,
        "keywords_total": kw_total,
        "keyword_ratio": round(kw_found / kw_total, 3) if kw_total else None,

        # Citation / grounding
        "citation_present": has_citation(answer),
        "boundary_compliant": (
            is_boundary_compliant(answer) if expected_type == "not_found" else None
        ),
        "hallucinated": detect_hallucination(state, expected_type),

        # Trace
        "final_answer": answer,
    }


def aggregate(results: list[dict]) -> dict[str, Any]:
    """Aggregate per-run metrics into a summary."""
    total = len(results)
    if total == 0:
        return {}

    def ratio(num: int, den: int) -> float:
        return round(num / den, 3) if den else 0.0

    tool_correct = sum(1 for r in results if r["tool_correct"])
    tool_total = sum(1 for r in results if r["tool_correct"] is not None)

    retrieval_hits = sum(1 for r in results if r["retrieval_hit"] is True)
    retrieval_total = sum(1 for r in results if r["retrieval_hit"] is not None)

    citation_count = sum(1 for r in results if r["citation_present"])

    boundary_ok = sum(1 for r in results if r["boundary_compliant"] is True)
    boundary_total = sum(1 for r in results if r["boundary_compliant"] is not None)

    hallucinated = sum(1 for r in results if r["hallucinated"] is True)
    hallucination_total = sum(1 for r in results if r["hallucinated"] is not None)

    failures = sum(1 for r in results if r["failure"])
    latencies = [r["latency_s"] for r in results]

    # Per-category
    by_category: dict[str, list[dict]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)

    category_summary = {
        cat: {
            "count": len(items),
            "tool_correct": ratio(
                sum(1 for r in items if r["tool_correct"]),
                sum(1 for r in items if r["tool_correct"] is not None),
            ),
            "failures": sum(1 for r in items if r["failure"]),
        }
        for cat, items in by_category.items()
    }

    return {
        "total_runs": total,
        "tool_accuracy": ratio(tool_correct, tool_total),
        "retrieval_hit_rate": ratio(retrieval_hits, retrieval_total),
        "citation_rate": ratio(citation_count, total),
        "boundary_compliance": ratio(boundary_ok, boundary_total),
        "hallucination_rate": ratio(hallucinated, hallucination_total),
        "failure_rate": ratio(failures, total),
        "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "max_latency_s": round(max(latencies), 2) if latencies else 0.0,
        "by_category": category_summary,
    }