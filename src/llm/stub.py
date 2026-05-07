from __future__ import annotations
import json
import re
import time
from typing import List

from .base import LLM, Message, LLMResponse


class StubLLM:
    """Deterministic, offline-friendly 'LLM' that drives the agent via simple rules.

    The agent prompts contain a JSON tool-call schema; this stub matches keywords in the
    user question to emit either a tool call or a final answer. Useful for CI, tests,
    and as a baseline in the eval table.
    """

    name = "stub"

    def chat(self, messages: List[Message], *, temperature: float = 0.0,
             max_tokens: int = 1024) -> LLMResponse:
        t0 = time.time()
        # find the original user question (the last user message that isn't a tool result)
        user_q = ""
        tool_results: list[str] = []
        for m in messages:
            if m.role == "user" and not m.content.startswith("TOOL_RESULT"):
                user_q = m.content
            if m.role == "user" and m.content.startswith("TOOL_RESULT"):
                tool_results.append(m.content)

        text = self._decide(user_q, tool_results)
        return LLMResponse(text=text, model=self.name, latency_s=time.time() - t0)

    def _decide(self, q: str, tool_results: list[str]) -> str:
        ql = q.lower()
        # If we have search results, produce a final answer with citations from them.
        if tool_results:
            cites = self._extract_citations(tool_results[-1])
            answer = self._compose_answer(q, tool_results[-1])
            return json.dumps({"final_answer": answer, "citations": cites})

        # First turn: choose a tool.
        m = re.search(r"calls\s+the\s+([A-Za-z_]\w*)\s+function", ql) or \
            re.search(r"what\s+calls\s+([A-Za-z_]\w*)", ql)
        if m:
            return json.dumps({"tool": "get_callers", "args": {"name": m.group(1)}})
        m = re.search(r"([A-Za-z_]\w*)\s+function\s+call", ql)
        if m:
            return json.dumps({"tool": "get_callees", "args": {"name": m.group(1)}})
        # GraphRAG: dependency/impact/related queries
        m = re.search(r"(?:dependenc|impact|related\s+to|connected\s+to|graph.*)\s+(?:of\s+)?([A-Za-z_]\w*)", ql)
        if m:
            return json.dumps({"tool": "graph_search", "args": {"names": [m.group(1)], "hops": 2}})
        # default: keyword search
        return json.dumps({"tool": "search", "args": {"query": q, "k": 6}})

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

    def _compose_answer(self, q: str, tool_block: str) -> str:
        # Pull the first matching symbol mentioned in the tool block as a hint.
        m = re.search(r"name=([A-Za-z_]\w*)", tool_block)
        sym = m.group(1) if m else "the relevant code"
        path_m = re.search(r"([\w./\-]+\.py):(\d+)-(\d+)", tool_block)
        loc = f" in {path_m.group(1)}" if path_m else ""
        return f"Based on the retrieved evidence, {sym}{loc} is the most relevant result for: {q.strip()}"
