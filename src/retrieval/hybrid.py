from __future__ import annotations
from typing import List

from .chunker import Chunk
from .bm25_index import BM25Index
from .vector_index import VectorIndex


class HybridRetriever:
    """Reciprocal-rank-fusion of BM25 and dense retrievers."""

    def __init__(self, bm25: BM25Index, vector: VectorIndex, *, rrf_k: int = 60):
        self.bm25 = bm25
        self.vector = vector
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 8) -> List[Chunk]:
        b = self.bm25.search(query, k=k * 2)
        v = self.vector.search(query, k=k * 2)
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for rank, (c, _) in enumerate(b):
            scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
            chunks[c.chunk_id] = c
        for rank, (c, _) in enumerate(v):
            scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
            chunks[c.chunk_id] = c
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [chunks[cid] for cid, _ in ranked]
