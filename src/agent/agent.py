from __future__ import annotations
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.config import SETTINGS
from src.index_repo import RepoIndex
from src.llm import LLM, Message
from .prompts import SYSTEM_PROMPT, FEWSHOT, GRAPH_SYSTEM_PROMPT, BASELINE_SYSTEM_PROMPT
from .tools import ToolBox, EvidenceSpan
from .memory import SessionMemory
from .logging import StructuredLogger
from .guardrails import check_query, scan_answer


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
    grounding_score: float = 0.0
    model: str = ""
    latency_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    def to_dict(self) -> dict:
        return {
            "question": self.question, "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "trace": self.trace, "grounded": self.grounded,
            "grounding_score": self.grounding_score,
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


def _extract_symbols(text: str) -> List[str]:
    """Extract likely symbol names (identifiers) from answer text."""
    return re.findall(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\b", text)


class CodeGraphAgent:
    def __init__(self, idx: RepoIndex, llm: LLM, *,
                 max_steps: int | None = None,
                 memory: Optional[SessionMemory] = None,
                 session_id: str = "default",
                 use_graph: bool = True):
        self.idx = idx
        self.llm = llm
        self.max_steps = max_steps or SETTINGS.max_agent_steps
        self.memory = memory
        self.session_id = session_id
        self.use_graph = use_graph

    def ask(self, question: str) -> AgentResult:
        t0 = time.time()
        toolbox = ToolBox(self.idx, use_graph=self.use_graph)
        model_name = getattr(self.llm, "name", "unknown")
        slog = StructuredLogger(session_id=self.session_id, model=model_name)

        # ---- Safety guardrail: pre-query filter ----
        refusal = check_query(question)
        if refusal:
            slog.log_guardrail(reason=refusal, blocked=True)
            slog.flush()
            return AgentResult(
                question=question,
                answer=(
                    "I'm unable to answer that question \u2014 it appears to "
                    "request sensitive or secret data."
                ),
                citations=[], trace=[{"step": 0, "guardrail": refusal}],
                grounded=False, grounding_score=0.0,
                model=model_name, latency_s=time.time() - t0,
            )

        # Build system prompt with session memory context if available
        base_prompt = GRAPH_SYSTEM_PROMPT if self.use_graph else BASELINE_SYSTEM_PROMPT
        system_content = base_prompt + "\n\n" + FEWSHOT
        if self.memory:
            prior_ctx = self.memory.retrieve_context(question)
            if prior_ctx:
                system_content += (
                    "\n\n--- SESSION MEMORY ---\n"
                    "Below are summaries of prior Q&A exchanges in this session. "
                    "Use them to understand context and chain follow-up questions.\n\n"
                    + prior_ctx
                )

        messages: List[Message] = [
            Message(role="system", content=system_content),
            Message(role="user", content=question),
        ]
        trace: List[Dict[str, Any]] = []
        tokens_in = tokens_out = 0
        retries = 0  # track re-retrieval attempts

        for step in range(self.max_steps):
            slog.start_step(step)
            resp = self.llm.chat(messages, temperature=0.0)
            tokens_in += resp.prompt_tokens
            tokens_out += resp.completion_tokens
            decision = _extract_json(resp.text)
            trace.append({"step": step, "raw": resp.text, "decision": decision})

            # Log the planner decision
            slog.log_planner(
                step=step, prompt_summary=question,
                raw_output=resp.text,
                tokens_in=resp.prompt_tokens, tokens_out=resp.completion_tokens,
            )

            if "final_answer" in decision:
                answer_text = str(decision.get("final_answer", "")).strip()
                raw_citations = decision.get("citations", [])
                citations = self._validate_citations(raw_citations, toolbox.seen)

                # Grounding verification
                grounding_score = self._verify_grounding(answer_text, citations)
                grounded = len(citations) > 0 and grounding_score > 0.0

                # Log critic check
                slog.log_critic(
                    step=step, n_raw=len(raw_citations),
                    n_validated=len(citations),
                    grounding_score=grounding_score,
                )

                # ---- Agentic re-retrieval: loop back if grounding is weak ----
                threshold = SETTINGS.grounding_threshold
                if (grounding_score < threshold
                        and retries < SETTINGS.max_retries
                        and step < self.max_steps - 1):
                    retries += 1
                    slog.log_re_retrieval(
                        step=step, grounding_score=grounding_score,
                        threshold=threshold, retry_num=retries,
                    )
                    retry_msg = (
                        f"CRITIC: Your citations scored {grounding_score:.2f} grounding "
                        f"(below threshold {threshold}). Your cited line ranges did not "
                        f"match your answer well. Please re-search with different queries "
                        f"or use read_lines to get better evidence, then produce a new "
                        f"final_answer with stronger citations."
                    )
                    messages.append(Message(role="assistant", content=resp.text))
                    messages.append(Message(role="user", content=retry_msg))
                    trace.append({"step": step, "re_retrieval": True,
                                  "grounding_score": grounding_score,
                                  "retry": retries})
                    continue  # go back to the ReAct loop

                # ---- Post-answer guardrail: redact leaked secrets ----
                answer_text, was_redacted = scan_answer(answer_text)
                if was_redacted:
                    slog.log_guardrail(reason="answer contained secret patterns",
                                       blocked=False)

                result = AgentResult(
                    question=question, answer=answer_text,
                    citations=citations, trace=trace, grounded=grounded,
                    grounding_score=grounding_score,
                    model=model_name, latency_s=time.time() - t0,
                    tokens_in=tokens_in, tokens_out=tokens_out,
                )

                # Log final answer
                slog.log_final_answer(
                    answer=answer_text, grounded=grounded,
                    grounding_score=grounding_score,
                    latency_ms=(time.time() - t0) * 1000,
                )
                slog.flush()

                # Save to session memory
                if self.memory:
                    symbols = _extract_symbols(answer_text)
                    self.memory.add(
                        question=question,
                        answer=answer_text,
                        citations=[c.to_dict() for c in citations],
                        symbols=symbols,
                    )

                return result

            tool = decision.get("tool")
            args = decision.get("args", {}) or {}
            if not tool:
                slog.log_error(step, "malformed JSON — no 'tool' or 'final_answer'")
                messages.append(Message(role="assistant", content=resp.text))
                messages.append(Message(role="user", content=(
                    "TOOL_RESULT error: response was not valid JSON with 'tool' or 'final_answer'. "
                    "Reply with the JSON schema described in the system prompt."
                )))
                continue

            # Log tool call
            slog.log_tool_call(step=step, tool_name=tool, tool_args=args)
            tool_t0 = time.time()
            tool_out = toolbox.call(tool, args)
            slog.log_tool_result(
                step=step, tool_name=tool, result=tool_out,
                latency_ms=(time.time() - tool_t0) * 1000,
            )
            messages.append(Message(role="assistant", content=resp.text))
            messages.append(Message(role="user", content=tool_out))

        # ran out of steps
        slog.log_error(step=self.max_steps - 1, error="max steps exceeded")
        slog.flush()
        return AgentResult(
            question=question, answer="I don't know.",
            citations=[], trace=trace, grounded=False,
            grounding_score=0.0,
            model=model_name, latency_s=time.time() - t0,
            tokens_in=tokens_in, tokens_out=tokens_out,
        )

    # ---- critic: drop citations that don't overlap any retrieved span ---------
    def _validate_citations(self, raw: list, seen: List[EvidenceSpan]) -> List[Citation]:
        out: List[Citation] = []
        for c in raw or []:
            try:
                fp = str(c["filepath"]).replace("\\", "/")
                rng = c.get("line_ranges") or [c.get("start_line"), c.get("end_line")]
                s, e = int(rng[0]), int(rng[1] if len(rng) > 1 else rng[0])
            except Exception:
                continue
            span = EvidenceSpan(fp, s, e)
            if any(span.overlaps(x) for x in seen):
                out.append(Citation(filepath=fp, line_ranges=[s, e]))
        return out

    # ---- grounding verification -----------------------------------------------
    def _verify_grounding(self, answer: str, citations: List[Citation]) -> float:
        """Re-fetch cited line ranges and check if key terms from the answer
        actually appear in the source code at those locations.

        Returns a grounding_score between 0.0 and 1.0:
          - For each citation, extract source text from the file
          - Tokenize salient words from the answer (length > 3)
          - Score = fraction of citations where >= 1 salient word matches the source
        """
        if not citations:
            return 0.0

        answer_lower = answer.lower()
        salient = {w for w in re.findall(r"[a-z_][a-z0-9_]{3,}", answer_lower)}
        if not salient:
            return 0.0

        verified = 0
        for cite in citations:
            try:
                source_text = self.idx.read_lines(
                    cite.filepath, cite.line_ranges[0], cite.line_ranges[1]
                ).lower()
                source_words = set(re.findall(r"[a-z_][a-z0-9_]{3,}", source_text))
                if salient & source_words:
                    verified += 1
            except Exception:
                continue

        return verified / len(citations) if citations else 0.0
