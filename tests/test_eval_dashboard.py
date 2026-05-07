from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import src.api.main as api_main
from src.eval.dataset import validate_eval_dataset
from src.eval.harness import format_multi_eval_results


def test_format_multi_eval_results_summary_shape():
    results = {
        "stub": {
            "model": "stub",
            "aggregate": {
                "n": 2,
                "answer_em": 0.5,
                "answer_contains": 1.0,
                "citation_grounded": 0.5,
                "citation_f1": 0.75,
                "hallucination_rate": 0.5,
                "avg_grounding_score": 0.6,
                "avg_latency_s": 0.2,
                "by_difficulty": {"easy": {"n": 1}},
                "by_category": {"lookup": {"n": 2}},
            },
            "cases": [{"case_id": "q1"}],
        },
        "openai:gpt-4o-mini": {
            "model": "openai:gpt-4o-mini",
            "aggregate": {
                "n": 2,
                "answer_em": 1.0,
                "answer_contains": 1.0,
                "citation_grounded": 1.0,
                "citation_f1": 1.0,
                "hallucination_rate": 0.0,
                "avg_grounding_score": 1.0,
                "avg_latency_s": 0.4,
                "by_difficulty": {"easy": {"n": 1}},
                "by_category": {"lookup": {"n": 2}},
            },
            "cases": [{"case_id": "q1"}],
        },
    }

    payload = format_multi_eval_results(results, models=["stub", "openai:gpt-4o-mini"])

    assert payload["models"] == ["stub", "openai:gpt-4o-mini"]
    assert [row["model"] for row in payload["summary_rows"]] == ["stub", "openai:gpt-4o-mini"]
    assert payload["summary_rows"][0]["answer_em"] == 0.5
    assert payload["aggregates_by_model"]["stub"]["by_difficulty"]["easy"]["n"] == 1
    assert payload["cases_by_model"]["openai:gpt-4o-mini"][0]["case_id"] == "q1"


def test_evaluate_single_model_payload(monkeypatch):
    def fake_run_eval(path, *, model, repo_sources, max_cases):
        assert path == Path("data/demo.json")
        assert model == "stub"
        assert repo_sources == {}
        assert max_cases == 3
        return {"model": model, "aggregate": {"n": 3}, "cases": [{"case_id": "q1"}]}

    monkeypatch.setattr(api_main, "run_eval", fake_run_eval)

    body = api_main.evaluate(api_main.EvalReq(**{
        "model": "stub",
        "eval_path": "data/demo.json",
        "max_cases": 3,
    }))

    assert body["model"] == "stub"
    assert body["aggregate"]["n"] == 3
    assert body["cases"][0]["case_id"] == "q1"


def test_evaluate_multi_model_payload(monkeypatch):
    def fake_run_eval_multi(path, *, models, repo_sources, max_cases):
        assert path == Path("data/demo.json")
        assert models == ["stub", "openai:gpt-4o-mini"]
        assert repo_sources == {}
        assert max_cases == 5
        return {
            "stub": {
                "model": "stub",
                "aggregate": {
                    "n": 5,
                    "answer_em": 0.4,
                    "answer_contains": 0.8,
                    "citation_grounded": 0.6,
                    "citation_f1": 0.7,
                    "hallucination_rate": 0.4,
                    "avg_grounding_score": 0.5,
                    "avg_latency_s": 0.1,
                    "by_difficulty": {},
                    "by_category": {},
                },
                "cases": [{"case_id": "q1", "answer_em": 0}],
            },
            "openai:gpt-4o-mini": {
                "model": "openai:gpt-4o-mini",
                "aggregate": {
                    "n": 5,
                    "answer_em": 0.8,
                    "answer_contains": 1.0,
                    "citation_grounded": 1.0,
                    "citation_f1": 0.9,
                    "hallucination_rate": 0.0,
                    "avg_grounding_score": 0.9,
                    "avg_latency_s": 0.2,
                    "by_difficulty": {},
                    "by_category": {},
                },
                "cases": [{"case_id": "q1", "answer_em": 1}],
            },
        }

    monkeypatch.setattr(api_main, "run_eval_multi", fake_run_eval_multi)

    body = api_main.evaluate(api_main.EvalReq(**{
        "models": ["stub", "openai:gpt-4o-mini"],
        "eval_path": "data/demo.json",
        "max_cases": 5,
    }))

    assert body["models"] == ["stub", "openai:gpt-4o-mini"]
    assert len(body["summary_rows"]) == 2
    assert body["summary_rows"][1]["model"] == "openai:gpt-4o-mini"
    assert body["cases_by_model"]["stub"][0]["case_id"] == "q1"


def test_evaluate_rejects_empty_models():
    with pytest.raises(ValidationError):
        api_main.EvalReq(models=[])


def test_eval_dataset_is_valid_and_balanced():
    stats = validate_eval_dataset(Path("data/CodeGraphEval_50.json"))

    assert stats["count"] == 50
    assert stats["category_counts"] == {
        "lookup": 14,
        "config-default": 10,
        "call-trace": 13,
        "dependency-impact": 13,
    }
    assert stats["difficulty_counts"] == {
        "easy": 15,
        "medium": 23,
        "hard": 12,
    }
