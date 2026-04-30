from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / ".codegraph_index"
INDEX_DIR.mkdir(exist_ok=True)


@dataclass
class Settings:
    embed_model: str = os.getenv("CG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embed_dim: int = int(os.getenv("CG_EMBED_DIM", "384"))
    top_k: int = int(os.getenv("CG_TOP_K", "8"))
    max_agent_steps: int = int(os.getenv("CG_MAX_STEPS", "6"))
    default_model: str = os.getenv("CG_DEFAULT_MODEL", "stub")


SETTINGS = Settings()
