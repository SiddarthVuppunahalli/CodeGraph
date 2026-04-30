from __future__ import annotations
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.config import SETTINGS
from src.index_repo import RepoIndex
from src.llm import LLM, Message
from .prompts import SYSTEM_PROMPT, FEWSHOT
from .tools import ToolBox, EvidenceSpan


@dataclass
class Citation:
    filepath: str
    line_ranges: List[int]  # [start, end]

    def to_dict(self) -> dict:
        return {"filepath": self.filepath, "line_ranges": list(self.line_ranges)}


@dataclass
class AgentResult:
    question: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    grounded: bool = False
    model: str = ""
    latency_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    def to_dict(self) -> dict:
        return {
            "question": self.question, "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "trace": self.trace, "grounded": self.grounded,
            "model": self.model, "latency_s": self.latency_s,
            "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
        }


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Dict[str, Any]:
    m = _JSON_RE.search(text)
    if not m:
        return {"final_answer": text.strip(), "citations": []}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"final_answer": text.strip(), "citations": []}


class CodeGraphAgent:
    def __init__(self, idx: RepoIndex, llm: LLM, *, max_steps: int | None = None):
        self.idx = idx
        self.llm = llm
        self.max_steps = max_steps or SETTINGS.max_agent_steps

    def ask(self, question: str) -> AgentResult:
        t0 = time.time()
        toolbox = ToolBox(self.idx)
        messages: List[Message] = [
            Message(role="system", content=SYSTEM_PROMPT + "\n\n" + FEWSHOT),
            Message(role="user", content=question),
        ]
        trace: List[Dict[str, Any]] = []
        tokens_in = tokens_out = 0
        model_name = getattr(self.llm, "name", "unknown")

        for step in range(self.max_steps):
            resp = self.llm.chat(messages, temperature=0.0)
            tokens_in += resp.prompt_tokens
            tokens_out += resp.completion_tokens
            decision = _extract_json(resp.text)
            trace.append({"step": step, "raw": resp.text, "decision": decision})

            if "final_answer" in decision:
                citations = self._validate_citations(decision.get("citations", []), toolbox.seen)
                grounded = len(citations) > 0
                return AgentResult(
                    question=question,
                    answer=str(decision.get("final_answer", "")).strip(),
                    citations=citations, trace=trace, grounded=grounded,
                    model=model_name, latency_s=time.time() - t0,
                    tokens_in=tokens_in, tokens_out=tokens_out,
                )

            tool = decision.get("tool")
            args = decision.get("args", {}) or {}
            if not tool:
                # malformed; nudge once and break to avoid loops
                messages.append(Message(role="assistant", content=resp.text))
                messages.append(Message(role="user", content=(
                    "TOOL_RESULT error: response was not valid JSON with 'tool' or 'final_answer'. "
                    "Reply with the JSON schema described in the system prompt."
                )))
                continue

            tool_out = toolbox.call(tool, args)
            messages.append(Message(role="assistant", content=resp.text))
            messages.append(Message(role="user", content=tool_out))

        # ran out of steps — return best effort
        return AgentResult(
            question=question,
            answer="I don't know.",
            citations=[], trace=trace, grounded=False,
            model=model_name, latency_s=time.time() - t0,
            tokens_in=tokens_in, tokens_out=tokens_out,
        )

    # ---- critic: drop citations that don't overlap any retrieved span ---------
    def _validate_citations(self, raw: list, seen: List[EvidenceSpan]) -> List[Citation]:
        out: List[Citation] = []
        for c in raw or []:
            try:
                fp = str(c["filepath"])
                rng = c.get("line_ranges") or [c.get("start_line"), c.get("end_line")]
                s, e = int(rng[0]), int(rng[1] if len(rng) > 1 else rng[0])
            except Exception:
                continue
            span = EvidenceSpan(fp, s, e)
            if any(span.overlaps(x) for x in seen):
                out.append(Citation(filepath=fp, line_ranges=[s, e]))
        return out
