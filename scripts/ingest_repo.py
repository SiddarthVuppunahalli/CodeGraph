"""CLI: python -m scripts.ingest_repo https://github.com/psf/requests"""
from __future__ import annotations
import argparse

from src.index_repo import build_index
from src.ingest import fetch_repo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="GitHub URL, local dir, or .zip")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rid, root = fetch_repo(args.source, force=args.force)
    print(f"repo_id={rid}  root={root}")
    idx = build_index(rid, root)
    print(f"files={len(idx.parsed)}  symbols={sum(len(p.symbols) for p in idx.parsed.values())}  "
          f"chunks={len(idx.chunks)}  graph_nodes={idx.graph.graph.number_of_nodes()}  "
          f"graph_edges={idx.graph.graph.number_of_edges()}")


if __name__ == "__main__":
    main()
