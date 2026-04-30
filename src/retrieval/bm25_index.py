from __future__ import annotations
import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from .chunker import Chunk

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokenize(text: str) -> List[str]:
    # split snake_case + camelCase + dots
    parts: List[str] = []
    for tok in _TOKEN.findall(text):
        # camelCase split
        sub = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", tok).split()
        for s in sub:
            parts.extend(s.lower().split("_"))
    return [p for p in parts if p]


class BM25Index:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.tokens = [_tokenize(c.symbol_name + " " + c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokens) if self.tokens else None

    def search(self, query: str, k: int = 8) -> List[Tuple[Chunk, float]]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in idx if scores[i] > 0]
