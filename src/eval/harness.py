from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from src.agent import CodeGraphAgent
from src.config import ROOT
from src.index_repo import RepoIndex, build_index
from src.ingest import fetch_repo
from src.llm import get_llm
from .metrics import CaseMetrics, aggregate, score_case


def _log(msg: str) -> None:
    print(f"[eval] {msg}", flush=True)


def _group_by_repo(cases: List[dict]) -> Dict[str, List[dict]]:
    g: Dict[str, List[dict]] = defaultdict(list)
    for c in cases:
        key = f"{c.get('repo_name','?')}@{c.get('version','?')}"
        g[key].append(c)
    return g


def _fallback_repo_root(eval_path: Path, repo_name: str) -> Path:
    if repo_name == "CodeGraph":
        return ROOT
    return Path(eval_path).parent


def format_multi_eval_results(results: Dict[str, dict], *, models: List[str]) -> dict:
    summary_rows: List[dict] = []
    aggregates_by_model: Dict[str, dict] = {}
    cases_by_model: Dict[str, List[dict]] = {}

    for model in models:
        payload = results[model]
        agg = payload["aggregate"]
        aggregates_by_model[model] = agg
        cases_by_model[model] = payload["cases"]
        summary_rows.append({
            "model": model,
            "n": agg.get("n", 0),
            "answer_em": agg.get("answer_em", 0.0),
            "answer_contains": agg.get("answer_contains", 0.0),
            "citation_grounded": agg.get("citation_grounded", 0.0),
            "citation_f1": agg.get("citation_f1", 0.0),
            "hallucination_rate": agg.get("hallucination_rate", 0.0),
            "avg_grounding_score": agg.get("avg_grounding_score", 0.0),
            "avg_latency_s": agg.get("avg_latency_s", 0.0),
        })

    return {
        "models": list(models),
        "summary_rows": summary_rows,
        "aggregates_by_model": aggregates_by_model,
        "cases_by_model": cases_by_model,
    }


def run_eval(
    eval_path: Path,
    *, model: str = "stub",
    repo_sources: Optional[Dict[str, str]] = None,
    max_cases: Optional[int] = None,
    use_graph: bool = True,
) -> dict:
    """Run CodeGraphEval-50 against a single LLM and return per-case + aggregate metrics.

    repo_sources maps "repo_name@version" → GitHub URL or local path. If a key is missing
    we fall back to using the eval data dir itself as the 'repo' so unit-style cases run.
    """
    cases = json.loads(Path(eval_path).read_text())
    if max_cases:
        cases = cases[:max_cases]
    repo_sources = repo_sources or {}
    _log(f"Starting eval for model={model} on {len(cases)} cases from {Path(eval_path).name}")
    llm = get_llm(model)

    per_case: List[CaseMetrics] = []
    indexes: Dict[str, RepoIndex] = {}
    total_cases = len(cases)
    completed = 0

    for key, group in _group_by_repo(cases).items():
        _log(f"[{model}] Preparing repo group {key} with {len(group)} cases")
        src = repo_sources.get(key) or repo_sources.get(group[0].get("repo_name", ""))
        if src is None:
            root = _fallback_repo_root(eval_path, group[0].get("repo_name", ""))
            indexes[key] = build_index(repo_id=key, root=root)
        else:
            rid, root = fetch_repo(src)
            indexes[key] = build_index(repo_id=rid, root=root)
        agent = CodeGraphAgent(indexes[key], llm, use_graph=use_graph)
        for group_idx, case in enumerate(group, 1):
            _log(
                f"[{model}] Case {completed + 1}/{total_cases} in {key}: "
                f"{case.get('question', '')[:90]}"
            )
            result = agent.ask(case["question"])
            per_case.append(score_case(case, result.to_dict()))
            completed += 1
            if group_idx == len(group):
                _log(f"[{model}] Finished repo group {key} ({completed}/{total_cases} cases done)")

    _log(f"Completed eval for model={model}")
    return {
        "model": model,
        "aggregate": aggregate(per_case),
        "cases": [c.__dict__ for c in per_case],
    }


def run_eval_multi(eval_path: Path, *, models: List[str], **kwargs) -> dict:
    _log(f"Starting multi-eval for {len(models)} models: {', '.join(models)}")
    results = {m: run_eval(eval_path, model=m, **kwargs) for m in models}
    _log("Completed multi-eval")
    return results

def run_ablation(eval_path: Path, *, model: str, **kwargs) -> dict:
    _log(f"Starting ablation study for {model} (Graph RAG vs Baseline RAG)")
    results = {
        "GraphRAG": run_eval(eval_path, model=model, use_graph=True, **kwargs),
        "BaselineRAG": run_eval(eval_path, model=model, use_graph=False, **kwargs),
    }
    _log("Completed ablation study")
    return format_multi_eval_results(results, models=["GraphRAG", "BaselineRAG"])
