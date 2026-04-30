from __future__ import annotations
import hashlib
import io
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import requests

from src.config import INDEX_DIR


def _repo_id(source: str) -> str:
    return hashlib.sha1(source.encode()).hexdigest()[:12]


def _from_github(url: str, dest: Path) -> None:
    # Try git clone first; fall back to codeload zip download.
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True, capture_output=True,
        )
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # https://github.com/<owner>/<repo>(.git)?
    cleaned = url.rstrip("/").removesuffix(".git")
    parts = cleaned.split("/")
    owner, repo = parts[-2], parts[-1]
    for branch in ("main", "master"):
        zurl = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
        r = requests.get(zurl, timeout=60)
        if r.ok:
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                zf.extractall(dest)
            # codeload nests under <repo>-<branch>/
            inner = next(dest.iterdir())
            for child in inner.iterdir():
                shutil.move(str(child), str(dest / child.name))
            inner.rmdir()
            return
    raise RuntimeError(f"Could not fetch {url}")


def _from_zip(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)


def fetch_repo(source: str, *, force: bool = False) -> tuple[str, Path]:
    """Fetch a repo from GitHub URL or local zip / directory. Returns (repo_id, path)."""
    rid = _repo_id(source)
    out = INDEX_DIR / rid / "src"
    if out.exists() and not force:
        return rid, out
    if out.exists():
        shutil.rmtree(out.parent)
    out.mkdir(parents=True)

    p = Path(source)
    if source.startswith("http://") or source.startswith("https://"):
        _from_github(source, out)
    elif p.is_dir():
        shutil.copytree(p, out, dirs_exist_ok=True)
    elif p.is_file() and p.suffix == ".zip":
        _from_zip(p, out)
    else:
        raise ValueError(f"Unsupported source: {source}")
    return rid, out
