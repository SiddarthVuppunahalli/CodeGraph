"""Build the full repo index: parse → chunk → graph → retrievers."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from src.config import SETTINGS
from src.graph import DependencyGraphBuilder
from src.ingest import walk_python_files
from src.parsing import PythonCodeParser, ParsedFile
from src.retrieval import BM25Index, Chunk, HybridRetriever, VectorIndex, chunk_parsed_file


@dataclass
class RepoIndex:
    repo_id: str
    root: Path
    parsed: Dict[str, ParsedFile]
    chunks: List[Chunk]
    bm25: BM25Index
    vector: VectorIndex
    retriever: HybridRetriever
    graph: DependencyGraphBuilder

    def read_lines(self, filepath: str, start: int, end: int) -> str:
        # Normalize path separators for cross-platform compatibility
        fp = filepath.replace("\\", "/")
        p = self.root / fp
        if not p.exists():
            return f"<file not found: {filepath}>"
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        s = max(start - 1, 0)
        e = min(end, len(lines))
        return "\n".join(f"{i+1:>5}: {lines[i]}" for i in range(s, e))


def build_index(repo_id: str, root: Path) -> RepoIndex:
    parser = PythonCodeParser()
    parsed: Dict[str, ParsedFile] = {}
    chunks: List[Chunk] = []
    graph = DependencyGraphBuilder()

    for file_path in walk_python_files(root):
        rel = file_path.relative_to(root).as_posix()
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        pf = parser.parse_file(rel, source)
        parsed[rel] = pf
        graph.add_symbols(pf.symbols)
        chunks.extend(chunk_parsed_file(pf))

    # add calls after every symbol is registered (so callee resolution works)
    for pf in parsed.values():
        graph.add_calls(pf.calls)

    bm25 = BM25Index(chunks)
    vector = VectorIndex(chunks, model_name=SETTINGS.embed_model, dim=SETTINGS.embed_dim)
    retriever = HybridRetriever(bm25, vector)

    return RepoIndex(
        repo_id=repo_id, root=root, parsed=parsed,
        chunks=chunks, bm25=bm25, vector=vector,
        retriever=retriever, graph=graph,
    )
