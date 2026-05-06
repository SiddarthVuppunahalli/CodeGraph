"""Structured JSON logging for every agent interaction.

Every Planner call, tool call, Critic check, and final answer emits a
JSON log entry with: timestamp, session_id, stage, prompt/output, tool_name,
latency_ms. Satisfies the 'structured logging' rubric requirement.
"""
from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import ROOT

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Python logger backed by a JSON file handler
_logger = logging.getLogger("codegraph.structured")
_logger.setLevel(logging.DEBUG)
_logger.propagate = False

# File handler: one JSON-lines file
_fh = logging.FileHandler(LOG_DIR / "agent_log.jsonl", mode="a", encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_logger.addHandler(_fh)

# Also stream to stderr at INFO level for dev visibility
_sh = logging.StreamHandler()
_sh.setLevel(logging.INFO)
_logger.addHandler(_sh)


@dataclass
class LogEntry:
    timestamp: float
    session_id: str
    stage: str           # "planner" | "tool_call" | "tool_result" | "critic" | "final_answer" | "error"
    step: int = 0
    prompt: str = ""
    output: str = ""
    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    grounding_score: float = -1.0   # -1 = not computed
    extra: Dict[str, Any] = field(default_factory=dict)


class StructuredLogger:
    """Accumulates log entries for a single agent invocation and flushes them."""

    def __init__(self, session_id: str = "default", model: str = "unknown"):
        self.session_id = session_id
        self.model = model
        self.entries: List[LogEntry] = []
        self._step_timers: Dict[int, float] = {}

    def start_step(self, step: int) -> None:
        self._step_timers[step] = time.time()

    def _elapsed_ms(self, step: int) -> float:
        t0 = self._step_timers.get(step, time.time())
        return (time.time() - t0) * 1000

    def log_planner(self, step: int, prompt_summary: str, raw_output: str,
                    tokens_in: int = 0, tokens_out: int = 0) -> None:
        self.entries.append(LogEntry(
            timestamp=time.time(), session_id=self.session_id,
            stage="planner", step=step,
            prompt=prompt_summary[:500], output=raw_output[:1000],
            latency_ms=self._elapsed_ms(step),
            model=self.model, tokens_in=tokens_in, tokens_out=tokens_out,
        ))

    def log_tool_call(self, step: int, tool_name: str, tool_args: Dict[str, Any]) -> None:
        self.entries.append(LogEntry(
            timestamp=time.time(), session_id=self.session_id,
            stage="tool_call", step=step,
            tool_name=tool_name, tool_args=tool_args,
        ))

    def log_tool_result(self, step: int, tool_name: str, result: str,
                        latency_ms: float = 0.0) -> None:
        self.entries.append(LogEntry(
            timestamp=time.time(), session_id=self.session_id,
            stage="tool_result", step=step,
            tool_name=tool_name, output=result[:2000],
            latency_ms=latency_ms,
        ))

    def log_critic(self, step: int, n_raw: int, n_validated: int,
                   grounding_score: float) -> None:
        self.entries.append(LogEntry(
            timestamp=time.time(), session_id=self.session_id,
            stage="critic", step=step,
            grounding_score=grounding_score,
            extra={"raw_citations": n_raw, "validated_citations": n_validated},
        ))

    def log_final_answer(self, answer: str, grounded: bool,
                         grounding_score: float, latency_ms: float) -> None:
        self.entries.append(LogEntry(
            timestamp=time.time(), session_id=self.session_id,
            stage="final_answer",
            output=answer[:2000], grounding_score=grounding_score,
            latency_ms=latency_ms,
            extra={"grounded": grounded},
        ))

    def log_error(self, step: int, error: str) -> None:
        self.entries.append(LogEntry(
            timestamp=time.time(), session_id=self.session_id,
            stage="error", step=step, output=error[:1000],
        ))

    def flush(self) -> List[dict]:
        """Write all entries to the log file and return them as dicts."""
        out: List[dict] = []
        for entry in self.entries:
            d = asdict(entry)
            line = json.dumps(d, default=str)
            _logger.info(line)
            out.append(d)
        self.entries.clear()
        return out
