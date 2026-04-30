from __future__ import annotations
from typing import Iterable, List, Dict, Tuple

import networkx as nx

from src.parsing import Symbol, CallEdge


class DependencyGraphBuilder:
    """NetworkX-backed call/contains graph keyed by symbol qualified names."""

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        # name -> list[qualified_name] for callee resolution
        self._by_name: Dict[str, List[str]] = {}

    # ---- construction ---------------------------------------------------------
    def add_symbols(self, symbols: Iterable[Symbol]) -> None:
        for s in symbols:
            self.graph.add_node(
                s.qualified_name, kind=s.kind, name=s.name, filepath=s.filepath,
                start_line=s.start_line, end_line=s.end_line,
                docstring=s.docstring,
            )
            self._by_name.setdefault(s.name, []).append(s.qualified_name)
            if s.parent and s.parent in self.graph:
                self.graph.add_edge(s.parent, s.qualified_name, type="contains")

    def add_calls(self, calls: Iterable[CallEdge]) -> None:
        for c in calls:
            for callee_qn in self._by_name.get(c.callee_name, []):
                self.graph.add_edge(
                    c.caller_qualified, callee_qn, type="calls",
                    filepath=c.filepath, line=c.line,
                )

    # ---- queries --------------------------------------------------------------
    def resolve(self, name: str) -> List[str]:
        """Map a bare name (e.g. 'authenticate_user') to qualified names."""
        return list(self._by_name.get(name, []))

    def get_callers(self, target: str) -> List[Tuple[str, dict]]:
        results: List[Tuple[str, dict]] = []
        candidates = [target] if target in self.graph else self.resolve(target)
        for qn in candidates:
            for pred in self.graph.predecessors(qn):
                edata = self.graph.get_edge_data(pred, qn) or {}
                if edata.get("type") == "calls":
                    results.append((pred, {**self.graph.nodes[pred], **edata}))
        return results

    def get_callees(self, target: str) -> List[Tuple[str, dict]]:
        results: List[Tuple[str, dict]] = []
        candidates = [target] if target in self.graph else self.resolve(target)
        for qn in candidates:
            for succ in self.graph.successors(qn):
                edata = self.graph.get_edge_data(qn, succ) or {}
                if edata.get("type") == "calls":
                    results.append((succ, {**self.graph.nodes[succ], **edata}))
        return results

    def neighbors_subgraph(self, target: str, radius: int = 1) -> nx.DiGraph:
        roots = [target] if target in self.graph else self.resolve(target)
        nodes = set(roots)
        frontier = set(roots)
        for _ in range(radius):
            nxt = set()
            for n in frontier:
                nxt.update(self.graph.predecessors(n))
                nxt.update(self.graph.successors(n))
            frontier = nxt - nodes
            nodes |= nxt
        return self.graph.subgraph(nodes).copy()

    # ---- legacy compat (used by old run_eval) ---------------------------------
    def add_call(self, caller: str, callee: str, filepath: str = "unknown") -> None:
        self.graph.add_edge(caller, callee, type="calls", filepath=filepath)
        self._by_name.setdefault(callee, []).append(callee)

    def build_dummy_graph(self) -> None:
        self.add_call("login_handler", "authenticate_user", filepath="auth.py")
