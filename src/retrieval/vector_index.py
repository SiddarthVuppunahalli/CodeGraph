from __future__ import annotations
import hashlib
from typing import List, Tuple

import numpy as np

try:
    import faiss
    _FAISS = True
except Exception:  # pragma: no cover
    _FAISS = False

from .chunker import Chunk
from .bm25_index import _tokenize


class _Embedder:
    """Sentence-transformers if installed, else a deterministic hashing fallback."""

    def __init__(self, model_name: str, dim: int):
        self.dim = dim
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.dim = self.model.get_sentence_embedding_dimension()
        except Exception:
            self.model = None

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.model is not None:
            v = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(v, dtype="float32")
        # hashing fallback: bag-of-tokens projected onto a fixed random basis (deterministic)
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in _tokenize(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 32) & 1 else -1.0
                out[i, idx] += sign
            n = np.linalg.norm(out[i]) or 1.0
            out[i] /= n
        return out


class VectorIndex:
    def __init__(self, chunks: List[Chunk], *, model_name: str, dim: int):
        self.chunks = chunks
        self.embedder = _Embedder(model_name, dim)
        if not chunks:
            self.index = None
            self.embeddings = None
            return
        self.embeddings = self.embedder.encode([c.text for c in chunks])
        self.dim = self.embeddings.shape[1]
        if _FAISS:
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(self.embeddings)
        else:
            self.index = None

    def search(self, query: str, k: int = 8) -> List[Tuple[Chunk, float]]:
        if self.embeddings is None:
            return []
        q = self.embedder.encode([query])
        if self.index is not None:
            scores, idxs = self.index.search(q, min(k, len(self.chunks)))
            return [(self.chunks[i], float(s)) for s, i in zip(scores[0], idxs[0]) if i >= 0]
        # pure-numpy fallback
        sims = (self.embeddings @ q[0]).astype("float32")
        idx = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in idx]
