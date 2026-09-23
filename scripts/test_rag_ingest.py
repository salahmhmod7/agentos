"""Test the full RAG ingest + search pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.rag.embedder import embedder
from app.agent.rag.ingest import ingest_file
from app.agent.rag.store import list_documents, search


def main() -> None:
    sample = Path("data/employee_handbook.txt")
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text(
        """Employee Handbook

Annual Leave Policy
Employees are entitled to 21 days of annual leave per year.
New employees receive annual leave after completing their probation period of 3 months.
Additional leave may be requested after 5 years of service.

Sick Leave Policy
Employees are entitled to 10 paid sick days per year.
Sick leave must be reported to the direct manager before 10 AM.
A doctor's note is required for absences longer than 3 days.

Working Hours
Standard working hours are 9 AM to 5 PM, Sunday to Thursday.
Remote work is allowed up to 2 days per week with manager approval.

Salary Review
Salaries are reviewed annually in January.
Performance bonuses are paid in March based on the previous year's evaluation.

Health Insurance
The company provides comprehensive health insurance for all employees.
Coverage includes dental, vision, and prescription medications.
Dependents can be added within 30 days of joining.

Training and Development
Employees have an annual training budget of $1000.
Online courses and conferences are both eligible for reimbursement.
Requests must be approved by the direct manager at least 2 weeks in advance.
""",
        encoding="utf-8",
    )

    print("=" * 70)
    print("INGEST (chunk_size=250, overlap=40)")
    print("=" * 70)
    summary = ingest_file(sample, chunk_size=250, overlap=40)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("DOCUMENTS IN STORE")
    print("=" * 70)
    for d in list_documents():
        print(f"  #{d['id']} {d['name']} ({d['num_chunks']} chunks)")

    print("\n" + "=" * 70)
    print("SEARCH TESTS")
    print("=" * 70)

    queries = [
        "How many vacation days do I get?",
        "What happens if I get sick?",
        "Can I work from home?",
        "When are bonuses paid?",
        "Do you cover dental?",
        "Can I attend a conference?",
    ]

    for q in queries:
        print(f"\n>>> Query: {q}")
        q_vec = embedder.embed_query(q)
        results = search(q_vec, top_k=2)
        for r in results:
            dist = r["distance"]
            text = r["text"][:120].replace("\n", " ")
            print(f"    [dist={dist:.3f}] chunk#{r['chunk_index']}: {text}...")


if __name__ == "__main__":
    main()