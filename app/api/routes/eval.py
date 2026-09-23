"""Evaluation results endpoint."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.eval.runner import LATEST_PATH


router = APIRouter(tags=["eval"])


@router.get("/eval/latest")
def latest_eval() -> dict:
    """Return the latest evaluation report (if any)."""
    p = Path(LATEST_PATH)
    if not p.exists():
        raise HTTPException(
            status_code=404,
            detail="No eval report yet. Run: python -m scripts.run_eval",
        )
    with p.open(encoding="utf-8") as f:
        return json.load(f)


@router.get("/eval/cases/{case_id}")
def get_case(case_id: str) -> dict:
    """Return a single case from the latest report."""
    p = Path(LATEST_PATH)
    if not p.exists():
        raise HTTPException(status_code=404, detail="No eval report yet")

    with p.open(encoding="utf-8") as f:
        report = json.load(f)

    for r in report.get("results", []):
        if r["id"] == case_id:
            return r

    raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")