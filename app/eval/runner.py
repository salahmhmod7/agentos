"""Evaluation runner — runs the dataset through the agent and writes a report."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app.agent.loop import run_agent
from app.agent.observability import EventType, Tracer, silent_handler
from app.eval.metrics import aggregate, evaluate_run


DATASET_PATH = Path("data/eval/dataset.json")
RESULTS_DIR = Path("data/eval")
LATEST_PATH = RESULTS_DIR / "latest_run.json"


def load_dataset(path: Path | None = None) -> dict:
    p = path or DATASET_PATH
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def run_one(item: dict, quiet: bool = True) -> dict:
    """Run a single test case and return its metrics.

    Uses a dedicated eval provider (usually Ollama) to avoid Groq rate limits.
    """
    from app.agent.llm import LLM, build_provider
    from app.core.config import settings

    # Use a dedicated provider for eval runs (Ollama by default)
    eval_llm = LLM(provider=build_provider(settings.eval_llm_provider))

    tracer = Tracer([silent_handler] if quiet else [])
    t0 = time.perf_counter()
    try:
        state, _ = run_agent(
            user_message=item["question"],
            llm=eval_llm,
            tracer=tracer,
        )
    except Exception as e:
        return {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "status": "exception",
            "iterations": 0,
            "latency_s": round(time.perf_counter() - t0, 2),
            "failure": True,
            "tools_used": [],
            "tool_correct": False,
            "retrieval_hit": None,
            "keywords_found": 0,
            "keywords_total": len(item.get("expected_keywords", []) or []),
            "keyword_ratio": 0.0,
            "citation_present": False,
            "boundary_compliant": None,
            "hallucinated": None,
            "final_answer": f"EXCEPTION: {e}",
        }

    latency = time.perf_counter() - t0
    return evaluate_run(item, state, latency)


def run_eval(
    quiet: bool = True,
    verbose: bool = True,
    inter_run_delay_s: float = 0.0,  # Ollama has no rate limit
) -> dict:
    """Run the full dataset and write a JSON report.

    inter_run_delay_s keeps us under Groq's free-tier rate limit (30 req/min).
    """
    dataset = load_dataset()
    cases = dataset["cases"]

    results: list[dict] = []

    if verbose:
        print(f"Running {len(cases)} cases (delay={inter_run_delay_s}s between runs)...\n")

    for i, item in enumerate(cases, 1):
        if verbose:
            print(f"[{i:2d}/{len(cases)}] {item['id']:4s} {item['category']:14s}", end="  ")
        r = run_one(item, quiet=quiet)
        results.append(r)
        if verbose:
            status = "✅" if (not r["failure"] and r.get("tool_correct", True)) else "❌"
            print(f"{status}  {r['latency_s']:5.1f}s")

        # Rate-limit guard: sleep between runs (except after the last one)
        if i < len(cases):
            time.sleep(inter_run_delay_s)

    summary = aggregate(results)
    # ... (rest unchanged)

    report = {
        "name": dataset.get("name", "eval"),
        "version": dataset.get("version", "1.0"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with LATEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if verbose:
        print_report(report)

    return report


def print_report(report: dict) -> None:
    s = report["summary"]
    print("\n" + "=" * 60)
    print(f"  EVAL REPORT — {report['name']} v{report['version']}")
    print("=" * 60)
    print(f"  Total runs          : {s['total_runs']}")
    print(f"  Tool accuracy       : {s['tool_accuracy']*100:.1f}%")
    print(f"  Retrieval hit rate  : {s['retrieval_hit_rate']*100:.1f}%")
    print(f"  Citation rate       : {s['citation_rate']*100:.1f}%")
    print(f"  Boundary compliance : {s['boundary_compliance']*100:.1f}%")
    print(f"  Hallucination rate  : {s['hallucination_rate']*100:.1f}%")
    print(f"  Failure rate        : {s['failure_rate']*100:.1f}%")
    print(f"  Avg latency         : {s['avg_latency_s']}s")
    print(f"  Max latency         : {s['max_latency_s']}s")
    print("-" * 60)
    print("  By category:")
    for cat, cs in s["by_category"].items():
        print(f"    {cat:16s} runs={cs['count']:2d}  tool_ok={cs['tool_correct']*100:5.1f}%  failures={cs['failures']}")
    print("=" * 60)
    print(f"  Report saved to: {LATEST_PATH}")