from __future__ import annotations
import os
import time
from typing import List

from .base import Message, LLMResponse


class AnthropicLLM:
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        import anthropic  # lazy
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.name = f"anthropic:{model}"
        self.model = model

    def chat(self, messages: List[Message], *, temperature: float = 0.0,
             max_tokens: int = 1024) -> LLMResponse:
        t0 = time.time()
        sys_parts = [m.content for m in messages if m.role == "system"]
        chat = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        resp = self.client.messages.create(
            model=self.model,
            system="\n\n".join(sys_parts) if sys_parts else None,
            messages=chat,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return LLMResponse(
            text=text, model=self.name, latency_s=time.time() - t0,
            prompt_tokens=getattr(resp.usage, "input_tokens", 0),
            completion_tokens=getattr(resp.usage, "output_tokens", 0),
        )
