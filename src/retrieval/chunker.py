from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List

from src.parsing import ParsedFile, Symbol


@dataclass
class Chunk:
    chunk_id: str
    filepath: str
    start_line: int
    end_line: int
    symbol_kind: str
    symbol_name: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _slice_lines(source: str, start: int, end: int) -> str:
    lines = source.splitlines()
    s = max(start - 1, 0)
    e = min(end, len(lines))
    return "\n".join(lines[s:e])


def chunk_parsed_file(pf: ParsedFile, *, window: int = 60) -> List[Chunk]:
    """One chunk per top-level function/class. Plus sliding windows for files with no symbols."""
    chunks: List[Chunk] = []
    top_symbols = [s for s in pf.symbols if s.kind in ("function", "class")]

    if top_symbols:
        for s in top_symbols:
            text = _slice_lines(pf.source, s.start_line, s.end_line)
            cid = f"{pf.filepath}:{s.start_line}-{s.end_line}"
            chunks.append(Chunk(
                chunk_id=cid, filepath=pf.filepath,
                start_line=s.start_line, end_line=s.end_line,
                symbol_kind=s.kind, symbol_name=s.name, text=text,
            ))
    # also emit a small head chunk so module-level constants/configs are searchable
    head_end = min(window, len(pf.source.splitlines()))
    if head_end:
        chunks.append(Chunk(
            chunk_id=f"{pf.filepath}:1-{head_end}",
            filepath=pf.filepath, start_line=1, end_line=head_end,
            symbol_kind="module", symbol_name=pf.filepath,
            text=_slice_lines(pf.source, 1, head_end),
        ))
    return chunks
