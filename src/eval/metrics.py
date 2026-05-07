from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class CaseMetrics:
    case_id: str
    category: str
    difficulty: str
    answer_em: int
    answer_contains: int
    citation_grounded: int
    citation_precision: float
    citation_recall: float
    citation_f1: float
    hallucinated: int
    grounding_score: float = 0.0
    latency_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _spans_overlap(a: tuple[str, int, int], b: tuple[str, int, int]) -> bool:
    return a[0].replace("\\", "/") == b[0].replace("\\", "/") and not (a[2] < b[1] or b[2] < a[1])


def _to_span(c: Dict[str, Any]) -> tuple[str, int, int]:
    fp = str(c.get("filepath", "")).replace("\\", "/")
    rng = c.get("line_ranges") or [c.get("start_line", 1), c.get("end_line", 1)]
    if len(rng) == 1:
        return (fp, int(rng[0]), int(rng[0]))
    return (fp, int(rng[0]), int(rng[1]))


def score_case(case: dict, agent_result: dict) -> CaseMetrics:
    expected = _normalize(case["ground_truth_answer"])
    got = _normalize(agent_result.get("answer", ""))
    em = int(expected == got)
    contains = int(any(tok and tok in got for tok in expected.split()) and len(got) > 0)
    # better contains: every salient noun-ish word in expected appears in got
    salient = [w for w in expected.split() if len(w) > 3]
    if salient:
        contains = int(sum(w in got for w in salient) / len(salient) >= 0.6)

    gt_spans = [_to_span(c) for c in case.get("ground_truth_evidence", [])]
    pred_spans = [_to_span(c) for c in agent_result.get("citations", [])]

    if pred_spans and gt_spans:
        tp = sum(1 for p in pred_spans if any(_spans_overlap(p, g) for g in gt_spans))
        precision = tp / len(pred_spans)
        recall = sum(1 for g in gt_spans if any(_spans_overlap(g, p) for p in pred_spans)) / len(gt_spans)
        grounded = int(tp > 0)
    else:
        precision = recall = 0.0
        grounded = 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    hallucinated = int(not grounded)

    return CaseMetrics(
        case_id=str(case.get("question", ""))[:80],
        category=case.get("category", "?"),
        difficulty=case.get("difficulty", "?"),
        answer_em=em, answer_contains=contains,
        citation_grounded=grounded,
        citation_precision=precision,
        citation_recall=recall,
        citation_f1=f1,
        hallucinated=hallucinated,
        grounding_score=float(agent_result.get("grounding_score", 0.0)),
        latency_s=float(agent_result.get("latency_s", 0.0)),
        tokens_in=int(agent_result.get("tokens_in", 0)),
        tokens_out=int(agent_result.get("tokens_out", 0)),
    )


def aggregate(cases: List[CaseMetrics]) -> Dict[str, Any]:
    if not cases:
        return {}
    n = len(cases)
    def m(attr: str) -> float:
        return sum(getattr(c, attr) for c in cases) / n

    result: Dict[str, Any] = {
        "n": n,
        "answer_em": m("answer_em"),
        "answer_contains": m("answer_contains"),
        "citation_grounded": m("citation_grounded"),
        "citation_precision": m("citation_precision"),
        "citation_recall": m("citation_recall"),
        "citation_f1": m("citation_f1"),
        "hallucination_rate": m("hallucinated"),
        "avg_grounding_score": m("grounding_score"),
        "avg_latency_s": m("latency_s"),
        "avg_tokens_in": m("tokens_in"),
        "avg_tokens_out": m("tokens_out"),
    }

    # Difficulty-stratified breakdown
    by_difficulty: Dict[str, List[CaseMetrics]] = {}
    for c in cases:
        by_difficulty.setdefault(c.difficulty, []).append(c)
    result["by_difficulty"] = {
        diff: _sub_aggregate(group) for diff, group in sorted(by_difficulty.items())
    }

    # Category-stratified breakdown
    by_category: Dict[str, List[CaseMetrics]] = {}
    for c in cases:
        by_category.setdefault(c.category, []).append(c)
    result["by_category"] = {
        cat: _sub_aggregate(group) for cat, group in sorted(by_category.items())
    }

    return result


def _sub_aggregate(cases: List[CaseMetrics]) -> Dict[str, Any]:
    """Lightweight aggregate for stratified sub-groups."""
    n = len(cases)
    if not n:
        return {}
    def m(attr: str) -> float:
        return sum(getattr(c, attr) for c in cases) / n
    return {
        "n": n,
        "answer_em": m("answer_em"),
        "answer_contains": m("answer_contains"),
        "citation_grounded": m("citation_grounded"),
        "citation_f1": m("citation_f1"),
        "hallucination_rate": m("hallucinated"),
        "avg_grounding_score": m("grounding_score"),
    }
