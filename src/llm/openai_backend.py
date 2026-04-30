from __future__ import annotations
import os
import time
from typing import List

from .base import Message, LLMResponse


class OpenAILLM:
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI  # lazy
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.name = f"openai:{model}"
        self.model = model

    def chat(self, messages: List[Message], *, temperature: float = 0.0,
             max_tokens: int = 1024) -> LLMResponse:
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature, max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content or ""
        u = resp.usage
        return LLMResponse(
            text=text, model=self.name, latency_s=time.time() - t0,
            prompt_tokens=getattr(u, "prompt_tokens", 0),
            completion_tokens=getattr(u, "completion_tokens", 0),
        )
