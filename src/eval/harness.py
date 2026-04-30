from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from src.agent import CodeGraphAgent
from src.index_repo import RepoIndex, build_index
from src.ingest import fetch_repo
from src.llm import get_llm
from .metrics import CaseMetrics, aggregate, score_case


def _group_by_repo(cases: List[dict]) -> Dict[str, List[dict]]:
    g: Dict[str, List[dict]] = defaultdict(list)
    for c in cases:
        key = f"{c.get('repo_name','?')}@{c.get('version','?')}"
        g[key].append(c)
    return g


def run_eval(
    eval_path: Path,
    *, model: str = "stub",
    repo_sources: Optional[Dict[str, str]] = None,
    max_cases: Optional[int] = None,
) -> dict:
    """Run CodeGraphEval-50 against a single LLM and return per-case + aggregate metrics.

    repo_sources maps "repo_name@version" → GitHub URL or local path. If a key is missing
    we fall back to using the eval data dir itself as the 'repo' so unit-style cases run.
    """
    cases = json.loads(Path(eval_path).read_text())
    if max_cases:
        cases = cases[:max_cases]
    repo_sources = repo_sources or {}
    llm = get_llm(model)

    per_case: List[CaseMetrics] = []
    indexes: Dict[str, RepoIndex] = {}

    for key, group in _group_by_repo(cases).items():
        src = repo_sources.get(key) or repo_sources.get(group[0].get("repo_name", ""))
        if src is None:
            # synthetic fallback: build an empty index rooted at the eval dir
            root = Path(eval_path).parent
            indexes[key] = build_index(repo_id=key, root=root)
        else:
            rid, root = fetch_repo(src)
            indexes[key] = build_index(repo_id=rid, root=root)
        agent = CodeGraphAgent(indexes[key], llm)
        for case in group:
            result = agent.ask(case["question"])
            per_case.append(score_case(case, result.to_dict()))

    return {
        "model": model,
        "aggregate": aggregate(per_case),
        "cases": [c.__dict__ for c in per_case],
    }


def run_eval_multi(eval_path: Path, *, models: List[str], **kwargs) -> dict:
    return {m: run_eval(eval_path, model=m, **kwargs) for m in models}
