from __future__ import annotations
from pathlib import Path
from typing import Iterator

SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".mypy_cache",
    ".pytest_cache", ".codegraph_index", "dist", "build", ".tox", ".idea", ".vscode",
}


def walk_python_files(root: Path) -> Iterator[Path]:
    for p in root.rglob("*.py"):
        rel_parts = p.relative_to(root).parts
        if any(part in SKIP_DIRS for part in rel_parts):
            continue
        if p.stat().st_size > 1_000_000:  # skip > 1MB
            continue
        yield p
