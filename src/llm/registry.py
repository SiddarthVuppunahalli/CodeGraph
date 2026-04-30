from __future__ import annotations
from typing import List

from .base import LLM
from .stub import StubLLM


def available_models() -> List[str]:
    return ["stub", "openai", "anthropic", "hf"]


def get_llm(name: str, **kwargs) -> LLM:
    """Factory: 'stub' | 'openai[:model]' | 'anthropic[:model]' | 'hf[:model]'."""
    if ":" in name:
        family, model = name.split(":", 1)
    else:
        family, model = name, None

    if family == "stub":
        return StubLLM()
    if family == "openai":
        from .openai_backend import OpenAILLM
        return OpenAILLM(model or "gpt-4o-mini", **kwargs)
    if family == "anthropic":
        from .anthropic_backend import AnthropicLLM
        return AnthropicLLM(model or "claude-haiku-4-5-20251001", **kwargs)
    if family == "hf":
        from .hf_backend import HFTransformersLLM
        return HFTransformersLLM(model or "Qwen/Qwen2.5-Coder-1.5B-Instruct", **kwargs)
    raise ValueError(f"Unknown model family: {family}")
