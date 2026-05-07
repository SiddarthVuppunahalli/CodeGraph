from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

from src.index_repo import RepoIndex


@dataclass
class EvidenceSpan:
    filepath: str
    start_line: int
    end_line: int

    def overlaps(self, other: "EvidenceSpan") -> bool:
        return (self.filepath.replace("\\", "/") == other.filepath.replace("\\", "/")
                and not (self.end_line < other.start_line or other.end_line < self.start_line))


class ToolBox:
    """Executes tool calls against a RepoIndex and accumulates seen evidence spans."""

    def __init__(self, idx: RepoIndex):
        self.idx = idx
        self.seen: List[EvidenceSpan] = []

    def call(self, name: str, args: Dict[str, Any]) -> str:
        if name == "search":
            return self._search(args.get("query", ""), int(args.get("k", 6)))
        if name == "get_callers":
            return self._callers(args.get("name", ""))
        if name == "get_callees":
            return self._callees(args.get("name", ""))
        if name == "read_lines":
            return self._read(args.get("filepath", ""), int(args.get("start", 1)), int(args.get("end", 1)))
        if name == "graph_search":
            names = args.get("names", [])
            if isinstance(names, str):
                names = [names]
            return self._graph_search(names, int(args.get("hops", 2)))
        return f"ERROR: unknown tool {name}"

    def _record(self, fp: str, s: int, e: int):
        self.seen.append(EvidenceSpan(fp, s, e))

    def _search(self, query: str, k: int) -> str:
        results = self.idx.retriever.search(query, k=k)
        if not results:
            return "TOOL_RESULT search: (no results)"
        lines = ["TOOL_RESULT search:"]
        for c in results:
            self._record(c.filepath, c.start_line, c.end_line)
            head = c.text.splitlines()[:8]
            lines.append(
                f"- {c.filepath}:{c.start_line}-{c.end_line} "
                f"kind={c.symbol_kind} name={c.symbol_name}\n  "
                + "\n  ".join(head)
            )
        return "\n".join(lines)

    def _callers(self, name: str) -> str:
        callers = self.idx.graph.get_callers(name)
        if not callers:
            return f"TOOL_RESULT get_callers({name}): (none)"
        lines = [f"TOOL_RESULT get_callers({name}):"]
        for qn, data in callers:
            fp = data.get("filepath", "?")
            sl = int(data.get("start_line", data.get("line", 1)))
            el = int(data.get("end_line", sl))
            self._record(fp, sl, el)
            lines.append(f"- {fp}:{sl}-{el} name={data.get('name', qn)}")
        return "\n".join(lines)

    def _callees(self, name: str) -> str:
        callees = self.idx.graph.get_callees(name)
        if not callees:
            return f"TOOL_RESULT get_callees({name}): (none)"
        lines = [f"TOOL_RESULT get_callees({name}):"]
        for qn, data in callees:
            fp = data.get("filepath", "?")
            sl = int(data.get("start_line", data.get("line", 1)))
            el = int(data.get("end_line", sl))
            self._record(fp, sl, el)
            lines.append(f"- {fp}:{sl}-{el} name={data.get('name', qn)}")
        return "\n".join(lines)

    def _read(self, filepath: str, start: int, end: int) -> str:
        body = self.idx.read_lines(filepath, start, end)
        self._record(filepath, start, end)
        return f"TOOL_RESULT read_lines({filepath}:{start}-{end}):\n{body}"

    def _graph_search(self, names: List[str], hops: int) -> str:
        results = self.idx.graph.graph_search(names, hops=hops, max_nodes=15)
        if not results:
            return f"TOOL_RESULT graph_search({names}): (no nodes found)"
        lines = [f"TOOL_RESULT graph_search({names}, hops={hops}):"]
        for qn, data in results:
            fp = data.get("filepath", "?")
            sl = int(data.get("start_line", 1))
            el = int(data.get("end_line", sl))
            self._record(fp, sl, el)
            kind = data.get("kind", "?")
            name = data.get("name", qn)
            edges = data.get("_edges", [])
            edge_desc = ""
            if edges:
                edge_parts = []
                for e in edges[:3]:
                    if "from" in e:
                        edge_parts.append(f"called-by:{e['from']}")
                    elif "to" in e:
                        edge_parts.append(f"calls:{e['to']}")
                edge_desc = " edges=[" + ", ".join(edge_parts) + "]"
            lines.append(f"- {fp}:{sl}-{el} kind={kind} name={name}{edge_desc}")
        return "\n".join(lines)
