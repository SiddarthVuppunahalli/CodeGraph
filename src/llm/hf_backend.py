from __future__ import annotations
import time
from typing import List

from .base import Message, LLMResponse


class HFTransformersLLM:
    """Open-weight model served via Hugging Face transformers (Llama / Qwen / DeepSeek-Coder)."""

    def __init__(self, model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct", device: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy
        import torch  # lazy
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model_obj = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
        )
        self.name = f"hf:{model}"

    def chat(self, messages: List[Message], *, temperature: float = 0.0,
             max_tokens: int = 1024) -> LLMResponse:
        import torch
        t0 = time.time()
        chat = [{"role": m.role, "content": m.content} for m in messages]
        prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True,
        )
        ids = self.tokenizer(prompt, return_tensors="pt").to(self.model_obj.device)
        with torch.no_grad():
            out = self.model_obj.generate(
                **ids, max_new_tokens=max_tokens,
                do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return LLMResponse(
            text=text, model=self.name, latency_s=time.time() - t0,
            prompt_tokens=int(ids["input_ids"].shape[1]),
            completion_tokens=int(out.shape[1] - ids["input_ids"].shape[1]),
        )
