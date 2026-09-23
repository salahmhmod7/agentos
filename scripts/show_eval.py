"""Pretty-print the latest eval report."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


REPORT = Path("data/eval/latest_run.json")


def main() -> None:
    if not REPORT.exists():
        print("No report yet. Run: python -m scripts.run_eval")
        return

    with REPORT.open(encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    failures = [r for r in results if r["failure"]]

    print("=" * 70)
    print(f"FAILURES ({len(failures)}/{len(results)})")
    print("=" * 70)

    for r in failures:
        print(f"\n[{r['id']}] {r['category']} | {r['latency_s']}s | status={r['status']}")
        print(f"  tools: {r['tools_used']}")
        print(f"  answer: {(r['final_answer'] or '')[:400]}")

    print("\n" + "=" * 70)
    print("BOUNDARY CASES (B01-B05, A01-A05)")
    print("=" * 70)
    for r in results:
        if r["category"] in ("boundary", "adversarial"):
            print(f"\n[{r['id']}] status={r['status']} | latency={r['latency_s']}s")
            print(f"  boundary_compliant: {r['boundary_compliant']}")
            print(f"  answer: {(r['final_answer'] or '')[:300]}")

    print("\n" + "=" * 70)
    print("CITATION CASES (C01-C03)")
    print("=" * 70)
    for r in results:
        if r["category"] == "citation":
            print(f"\n[{r['id']}] status={r['status']} | citation_present={r['citation_present']}")
            print(f"  answer: {(r['final_answer'] or '')[:400]}")


if __name__ == "__main__":
    main()