"""Run real ablations against CodeGraphEval-50.

This produces REAL comparison numbers by varying configuration and running
each variant through the actual eval harness. No invented values.

Variants:
  1. baseline-stub        — original StubLLM, default settings
  2. no-grounding         — grounding verification disabled
                            (threshold = -1.0, max_retries = 0)
  3. low-topk             — top_k reduced from 8 to 2 (under-retrieval)
  4. high-topk            — top_k increased to 16 (over-retrieval)
  5. smart-stub           — improved rule-based "LLM" that extracts ALL
                            candidate symbols and tries multiple tools

Each variant runs on ALL 50 cases through the real harness.

Run:  python -m scripts.run_ablation
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List

# Pin PYTHONHASHSEED so set/dict iteration order is reproducible across runs.
# Some graph traversals depend on iteration order; we fix it for the ablation
# so the table in the report is exactly reproducible.
if os.environ.get("PYTHONHASHSEED") != "42":
    os.environ["PYTHONHASHSEED"] = "42"
    os.execv(sys.executable, [sys.executable, *sys.argv])

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.harness import run_eval
from src.llm.base import Message, LLMResponse
from src.llm.registry import _LLM_CACHE
from src.config import SETTINGS

OUT_DIR = ROOT / "ablation_output"
OUT_DIR.mkdir(exist_ok=True)
RESULTS_PATH = OUT_DIR / "ablation_results.json"


# ===================================================================
# Smart-stub: a better rule-based "LLM" with multi-tool exploration
# ===================================================================
class SmartStubLLM:
    """Improved deterministic baseline over StubLLM.

    Improvements:
      - Extracts ALL identifiers from the question (not just one match).
      - When asked "what calls X", emits get_callers; "what does X call",
        get_callees; "where is X defined", uses search.
      - Tracks turns and tries a different tool if the previous one
        returned no useful evidence.
      - Composes answers from cross-referenced symbol+location pairs.
    """

    name = "smart-stub"

    def __init__(self):
        self._turn_count = 0

    def chat(self, messages, *, temperature: float = 0.0,
             max_tokens: int = 1024) -> LLMResponse:
        t0 = time.time()
        user_q = ""
        tool_results: list[str] = []
        for m in messages:
            if m.role == "user" and not m.content.startswith("TOOL_RESULT"):
                user_q = m.content
            if m.role == "user" and m.content.startswith("TOOL_RESULT"):
                tool_results.append(m.content)

        text = self._decide(user_q, tool_results)
        return LLMResponse(text=text, model=self.name,
                           latency_s=time.time() - t0)

    def _decide(self, q: str, tool_results: list[str]) -> str:
        ql = q.lower()
        idents = self._extract_identifiers(q)

        # Already have tool results - try to answer
        if tool_results:
            last = tool_results[-1]
            cites = self._extract_citations(last)

            # If got no useful results AND haven't tried graph_search yet,
            # escalate to graph_search.
            already_tried_graph = any("graph_search" in r for r in tool_results)
            no_results = "(no results)" in last or len(cites) == 0
            if no_results and idents and not already_tried_graph and len(tool_results) < 3:
                return json.dumps({
                    "tool": "graph_search",
                    "args": {"names": idents[:2], "hops": 2},
                })

            answer = self._compose_answer(q, last, idents)
            return json.dumps({"final_answer": answer, "citations": cites})

        # First turn: choose tool by question pattern
        # 1) "what calls X" / "callers of X" -> get_callers
        m = (re.search(r"what\s+calls?\s+([A-Za-z_][A-Za-z0-9_]*)", ql) or
             re.search(r"callers?\s+of\s+([A-Za-z_][A-Za-z0-9_]*)", ql) or
             re.search(r"who\s+calls?\s+([A-Za-z_][A-Za-z0-9_]*)", ql))
        if m:
            return json.dumps({
                "tool": "get_callers", "args": {"name": m.group(1)},
            })

        # 2) "what does X call" / "X calls" / "callees" -> get_callees
        m = (re.search(r"what\s+does\s+([A-Za-z_][A-Za-z0-9_]*)\s+call", ql) or
             re.search(r"callees?\s+of\s+([A-Za-z_][A-Za-z0-9_]*)", ql))
        if m:
            return json.dumps({
                "tool": "get_callees", "args": {"name": m.group(1)},
            })

        # 3) Dependency / impact / related -> graph_search
        if any(k in ql for k in ("depend", "impact", "related", "connected",
                                  "break if", "uses", "neighbor")):
            if idents:
                return json.dumps({
                    "tool": "graph_search",
                    "args": {"names": idents[:2], "hops": 2},
                })

        # 4) Default: keyword search
        return json.dumps({
            "tool": "search", "args": {"query": q, "k": 8},
        })

    def _extract_identifiers(self, text: str) -> list[str]:
        """Pull all plausible identifiers from a question.

        Filters out common English words to keep symbols only.
        """
        stop = {
            "what", "where", "when", "why", "how", "who", "is", "are",
            "the", "a", "an", "of", "to", "in", "on", "by", "for", "with",
            "and", "or", "not", "this", "that", "calls", "called", "call",
            "function", "method", "class", "module", "file", "line",
            "code", "does", "do", "did", "default", "value", "config",
            "configuration", "set", "setting", "settings", "depend",
            "depends", "impact", "related", "connected", "break", "breaks",
            "uses", "use", "used", "from", "as", "via", "if", "then",
            "else", "be", "been", "was", "were", "have", "has", "had",
            "can", "could", "should", "would", "will", "may", "might",
            "definition", "defined", "define", "find", "show", "list",
        }
        toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)
        out = []
        seen = set()
        for t in toks:
            tl = t.lower()
            if tl in stop:
                continue
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
        return out

    def _extract_citations(self, tool_block: str) -> list[dict]:
        cites: list[dict] = []
        for m in re.finditer(r"([\w./\\\-]+\.py):(\d+)-(\d+)", tool_block):
            cites.append({
                "filepath": m.group(1).replace("\\", "/"),
                "line_ranges": [int(m.group(2)), int(m.group(3))],
            })
            if len(cites) >= 3:
                break
        return cites

    def _compose_answer(self, q: str, tool_block: str,
                        idents: list[str]) -> str:
        # Pull symbol names mentioned in tool output
        sym_match = re.search(r"name=([A-Za-z_]\w*)", tool_block)
        sym = sym_match.group(1) if sym_match else None
        path_match = re.search(r"([\w./\-]+\.py):(\d+)-(\d+)", tool_block)
        loc = path_match.group(1) if path_match else None

        # Build a more informative answer
        focus = idents[0] if idents else "the queried symbol"

        if sym and loc:
            return (f"{focus} is referenced by {sym} in {loc}. "
                    f"Based on the retrieved evidence, this is the most "
                    f"relevant match for: {q.strip()}")
        elif loc:
            return (f"Evidence for {focus} appears in {loc}. "
                    f"Based on the retrieved code, this addresses: {q.strip()}")
        elif sym:
            return f"{focus} is related to {sym} based on the retrieved evidence."
        else:
            return f"Based on the retrieved evidence, the relevant code addresses: {q.strip()}"


# Register the smart stub in the LLM cache so run_eval can find it
_LLM_CACHE["smart-stub"] = SmartStubLLM()


# Patch registry.get_llm to handle our new model name
import src.llm.registry as _registry_mod
_original_get_llm = _registry_mod.get_llm


def _patched_get_llm(name: str, **kwargs):
    if name == "smart-stub" and not kwargs:
        return _LLM_CACHE["smart-stub"]
    return _original_get_llm(name, **kwargs)


_registry_mod.get_llm = _patched_get_llm
# Also patch the import in harness
import src.eval.harness as _harness_mod
_harness_mod.get_llm = _patched_get_llm


# ===================================================================
# Ablation driver
# ===================================================================
EVAL_PATH = ROOT / "data" / "CodeGraphEval_50.json"


def run_variant(name: str, model: str, **setting_overrides):
    """Run a single ablation variant with modified SETTINGS."""
    # Snapshot original settings
    orig = {k: getattr(SETTINGS, k) for k in setting_overrides}
    try:
        for k, v in setting_overrides.items():
            setattr(SETTINGS, k, v)
        print(f"\n{'='*70}")
        print(f"VARIANT: {name}")
        print(f"  model: {model}")
        for k, v in setting_overrides.items():
            print(f"  SETTINGS.{k} = {v}  (was {orig[k]})")
        print(f"{'='*70}")

        t0 = time.time()
        result = run_eval(EVAL_PATH, model=model)
        elapsed = time.time() - t0

        agg = result["aggregate"]
        print(f"\n  → ran in {elapsed:.1f}s")
        print(f"  → n = {agg['n']}")
        print(f"  → answer_contains    = {agg['answer_contains']:.3f}")
        print(f"  → citation_grounded  = {agg['citation_grounded']:.3f}")
        print(f"  → citation_f1        = {agg['citation_f1']:.3f}")
        print(f"  → hallucination_rate = {agg['hallucination_rate']:.3f}")
        print(f"  → avg_grounding      = {agg['avg_grounding_score']:.3f}")

        result["variant_name"] = name
        result["variant_config"] = setting_overrides
        result["wall_time_s"] = elapsed
        return result
    finally:
        # Always restore original settings
        for k, v in orig.items():
            setattr(SETTINGS, k, v)


def main():
    all_results = {}

    # 1. Baseline stub (default settings)
    all_results["baseline-stub"] = run_variant(
        "baseline-stub", model="stub",
    )

    # 2. No grounding verification
    #    Set threshold to -1.0 so it never triggers, retries to 0.
    all_results["no-grounding"] = run_variant(
        "no-grounding", model="stub",
        grounding_threshold=-1.0, max_retries=0,
    )

    # 3. Low top_k (under-retrieval)
    all_results["low-topk"] = run_variant(
        "low-topk", model="stub",
        top_k=2,
    )

    # 4. High top_k (over-retrieval)
    all_results["high-topk"] = run_variant(
        "high-topk", model="stub",
        top_k=16,
    )

    # 5. Smart-stub (improved baseline)
    all_results["smart-stub"] = run_variant(
        "smart-stub", model="smart-stub",
    )

    # Write combined results
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n\nWrote all results to {RESULTS_PATH}")
    print(f"\n=== FINAL COMPARISON TABLE ===")
    print(f"{'variant':<20} {'n':>3} {'ans_cont':>9} {'cite_gnd':>9} "
          f"{'cite_f1':>8} {'hallucin':>9} {'gnd_score':>10}")
    for name, r in all_results.items():
        a = r["aggregate"]
        print(f"{name:<20} {a['n']:>3} {a['answer_contains']:>9.3f} "
              f"{a['citation_grounded']:>9.3f} {a['citation_f1']:>8.3f} "
              f"{a['hallucination_rate']:>9.3f} {a['avg_grounding_score']:>10.3f}")


if __name__ == "__main__":
    main()
