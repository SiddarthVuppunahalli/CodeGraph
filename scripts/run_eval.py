"""CLI: python -m scripts.run_eval --model stub --eval data/CodeGraphEval_50_sample.json"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from src.eval import run_eval
from src.eval.harness import run_eval_multi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/CodeGraphEval_50_sample.json")
    ap.add_argument("--model", default="stub", help="single model id, e.g. 'stub', 'openai:gpt-4o-mini'")
    ap.add_argument("--models", nargs="*", help="run multiple models for the comparison table")
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--repo-sources", default=None,
                    help="JSON dict mapping 'repo_name@version' to GitHub URL or local path")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo_sources = json.loads(Path(args.repo_sources).read_text()) if args.repo_sources else {}

    if args.models:
        result = run_eval_multi(
            Path(args.eval), models=args.models,
            repo_sources=repo_sources, max_cases=args.max_cases,
        )
        # tabular summary
        print(f"\n{'model':<40} {'EM':>5} {'CON':>5} {'GND':>5} {'F1':>5} {'HAL':>5} {'sec':>6}")
        for m, r in result.items():
            a = r["aggregate"]
            print(f"{m:<40} {a['answer_em']:.2f} {a['answer_contains']:.2f} "
                  f"{a['citation_grounded']:.2f} {a['citation_f1']:.2f} "
                  f"{a['hallucination_rate']:.2f} {a['avg_latency_s']:.2f}")
    else:
        result = run_eval(
            Path(args.eval), model=args.model,
            repo_sources=repo_sources, max_cases=args.max_cases,
        )
        print(json.dumps(result["aggregate"], indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
