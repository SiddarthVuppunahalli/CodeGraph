from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Protocol


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    extra: dict = field(default_factory=dict)


class LLM(Protocol):
    name: str
    def chat(self, messages: List[Message], *, temperature: float = 0.0,
             max_tokens: int = 1024) -> LLMResponse: ...
