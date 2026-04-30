from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from tree_sitter import Parser, Language
    import tree_sitter_python as tspython
    _TS_AVAILABLE = True
except Exception:  # pragma: no cover
    _TS_AVAILABLE = False


@dataclass
class Symbol:
    kind: str  # function | class | method | module
    name: str
    qualified_name: str  # "<relpath>::<owner>.<name>"
    filepath: str
    start_line: int  # 1-indexed inclusive
    end_line: int    # 1-indexed inclusive
    parent: Optional[str] = None
    docstring: str = ""


@dataclass
class CallEdge:
    caller_qualified: str
    callee_name: str
    filepath: str
    line: int


@dataclass
class ParsedFile:
    filepath: str
    source: str
    symbols: List[Symbol] = field(default_factory=list)
    calls: List[CallEdge] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)


class PythonCodeParser:
    """Tree-sitter Python parser. Falls back to a regex-lite parser if tree-sitter is missing."""

    def __init__(self) -> None:
        if _TS_AVAILABLE:
            try:
                self.lang = Language(tspython.language())
            except Exception:
                self.lang = tspython.language()  # already a Language object on some versions
            try:
                self.parser = Parser(self.lang)
            except TypeError:
                self.parser = Parser()
                self.parser.set_language(self.lang)
        else:
            self.parser = None

    def parse_file(self, filepath: str, source: str) -> ParsedFile:
        if self.parser is None:
            return self._fallback(filepath, source)
        tree = self.parser.parse(source.encode("utf-8"))
        pf = ParsedFile(filepath=filepath, source=source)
        self._walk(tree.root_node, source.encode("utf-8"), pf, owner=None)
        return pf

    def _node_text(self, node, src: bytes) -> str:
        return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def _docstring(self, body_node, src: bytes) -> str:
        if body_node is None:
            return ""
        for child in body_node.children:
            if child.type == "expression_statement" and child.child_count > 0:
                first = child.children[0]
                if first.type == "string":
                    return self._node_text(first, src).strip("'\" \n")
            break
        return ""

    def _walk(self, node, src: bytes, pf: ParsedFile, owner: Optional[str]):
        t = node.type
        if t == "function_definition":
            name_node = node.child_by_field_name("name")
            body_node = node.child_by_field_name("body")
            if name_node is not None:
                name = self._node_text(name_node, src)
                qn = f"{pf.filepath}::{owner + '.' if owner else ''}{name}"
                kind = "method" if owner else "function"
                pf.symbols.append(Symbol(
                    kind=kind, name=name, qualified_name=qn, filepath=pf.filepath,
                    start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1,
                    parent=owner, docstring=self._docstring(body_node, src),
                ))
                if body_node is not None:
                    self._collect_calls(body_node, src, pf, qn)
                    for child in body_node.children:
                        self._walk(child, src, pf, owner=owner)
            return
        if t == "class_definition":
            name_node = node.child_by_field_name("name")
            body_node = node.child_by_field_name("body")
            if name_node is not None:
                name = self._node_text(name_node, src)
                qn = f"{pf.filepath}::{name}"
                pf.symbols.append(Symbol(
                    kind="class", name=name, qualified_name=qn, filepath=pf.filepath,
                    start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1,
                    parent=owner, docstring=self._docstring(body_node, src),
                ))
                if body_node is not None:
                    for child in body_node.children:
                        self._walk(child, src, pf, owner=name)
            return
        if t in ("import_statement", "import_from_statement"):
            pf.imports.append(self._node_text(node, src).strip())
            return
        for child in node.children:
            self._walk(child, src, pf, owner)

    def _collect_calls(self, node, src: bytes, pf: ParsedFile, caller_qn: str):
        if node.type == "call":
            fn = node.child_by_field_name("function")
            if fn is not None:
                callee = self._node_text(fn, src).split(".")[-1].strip()
                if callee.isidentifier():
                    pf.calls.append(CallEdge(
                        caller_qualified=caller_qn, callee_name=callee,
                        filepath=pf.filepath, line=node.start_point[0] + 1,
                    ))
        for child in node.children:
            self._collect_calls(child, src, pf, caller_qn)

    def _fallback(self, filepath: str, source: str) -> ParsedFile:
        import re
        pf = ParsedFile(filepath=filepath, source=source)
        for i, line in enumerate(source.splitlines(), 1):
            m = re.match(r"\s*def\s+([A-Za-z_]\w*)\s*\(", line)
            if m:
                name = m.group(1)
                pf.symbols.append(Symbol(
                    kind="function", name=name,
                    qualified_name=f"{filepath}::{name}",
                    filepath=filepath, start_line=i, end_line=i,
                ))
            m = re.match(r"\s*class\s+([A-Za-z_]\w*)", line)
            if m:
                name = m.group(1)
                pf.symbols.append(Symbol(
                    kind="class", name=name,
                    qualified_name=f"{filepath}::{name}",
                    filepath=filepath, start_line=i, end_line=i,
                ))
            for callee in re.findall(r"([A-Za-z_]\w*)\s*\(", line):
                pf.calls.append(CallEdge(
                    caller_qualified=f"{filepath}::<module>",
                    callee_name=callee, filepath=filepath, line=i,
                ))
        return pf

    # backwards-compatible helpers (kept for older scripts)
    def extract_functions(self, source_code: bytes) -> List[str]:
        pf = self.parse_file("<inline>", source_code.decode("utf-8", errors="replace"))
        return [s.name for s in pf.symbols if s.kind in ("function", "method")]

    def extract_calls(self, source_code: bytes):
        pf = self.parse_file("<inline>", source_code.decode("utf-8", errors="replace"))
        return [(c.caller_qualified.split("::")[-1], c.callee_name) for c in pf.calls]
