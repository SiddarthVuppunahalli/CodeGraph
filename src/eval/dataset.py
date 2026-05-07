from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from src.config import ROOT

ALLOWED_CATEGORIES = {"lookup", "config-default", "call-trace", "dependency-impact"}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}


def _repo_root_for_case(case: Dict[str, Any], eval_path: Path) -> Path:
    if case.get("repo_name") == "CodeGraph":
        return ROOT
    return eval_path.parent


def validate_eval_dataset(eval_path: Path) -> Dict[str, Any]:
    path = Path(eval_path)
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("Eval dataset must be a JSON array.")

    seen_questions = set()
    category_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()

    for idx, case in enumerate(cases, 1):
        missing = [
            key for key in (
                "repo_name", "version", "question", "category",
                "ground_truth_answer", "ground_truth_evidence", "difficulty",
            )
            if key not in case
        ]
        if missing:
            raise ValueError(f"Case {idx} missing required fields: {missing}")

        question = str(case["question"]).strip()
        if not question:
            raise ValueError(f"Case {idx} has an empty question.")
        if question in seen_questions:
            raise ValueError(f"Duplicate question in case {idx}: {question}")
        seen_questions.add(question)

        category = str(case["category"])
        difficulty = str(case["difficulty"])
        if category not in ALLOWED_CATEGORIES:
            raise ValueError(f"Case {idx} has invalid category: {category}")
        if difficulty not in ALLOWED_DIFFICULTIES:
            raise ValueError(f"Case {idx} has invalid difficulty: {difficulty}")
        category_counts[category] += 1
        difficulty_counts[difficulty] += 1

        evidence = case["ground_truth_evidence"]
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"Case {idx} must have at least one evidence span.")

        repo_root = _repo_root_for_case(case, path)
        for ev in evidence:
            filepath = str(ev.get("filepath", "")).replace("\\", "/")
            line_ranges = ev.get("line_ranges")
            if not filepath or not isinstance(line_ranges, list) or len(line_ranges) != 2:
                raise ValueError(f"Case {idx} has malformed evidence: {ev}")

            start, end = int(line_ranges[0]), int(line_ranges[1])
            if start < 1 or end < start:
                raise ValueError(f"Case {idx} has invalid line range: {line_ranges}")

            target = repo_root / filepath
            if not target.exists():
                raise ValueError(f"Case {idx} references missing file: {filepath}")

            n_lines = len(target.read_text(encoding="utf-8", errors="replace").splitlines())
            if end > n_lines:
                raise ValueError(
                    f"Case {idx} line range {start}-{end} exceeds file length {n_lines} for {filepath}"
                )

    return {
        "count": len(cases),
        "category_counts": dict(category_counts),
        "difficulty_counts": dict(difficulty_counts),
    }
