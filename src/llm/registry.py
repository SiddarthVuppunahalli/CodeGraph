from __future__ import annotations
from typing import List

from .base import LLM
from .stub import StubLLM

_LLM_CACHE: dict[str, LLM] = {}


def available_models() -> List[str]:
    return ["stub", "openai", "anthropic", "hf"]


def get_llm(name: str, **kwargs) -> LLM:
    """Factory: 'stub' | 'openai[:model]' | 'anthropic[:model]' | 'hf[:model]'."""
    if not kwargs and name in _LLM_CACHE:
        return _LLM_CACHE[name]

    if ":" in name:
        family, model = name.split(":", 1)
    else:
        family, model = name, None

    llm: LLM
    if family == "stub":
        llm = StubLLM()
    elif family == "openai":
        from .openai_backend import OpenAILLM
        llm = OpenAILLM(model or "gpt-4o-mini", **kwargs)
    elif family == "anthropic":
        from .anthropic_backend import AnthropicLLM
        llm = AnthropicLLM(model or "claude-haiku-4-5-20251001", **kwargs)
    elif family == "hf":
        from .hf_backend import HFTransformersLLM
        llm = HFTransformersLLM(model or "Qwen/Qwen2.5-Coder-1.5B-Instruct", **kwargs)
    else:
        raise ValueError(f"Unknown model family: {family}")

    if not kwargs:
        _LLM_CACHE[name] = llm
    return llm
