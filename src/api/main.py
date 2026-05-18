from __future__ import annotations
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from src.agent import CodeGraphAgent, SESSION_STORE
from src.config import DATA_DIR
from src.eval import run_eval, run_eval_multi, format_multi_eval_results, run_ablation
from src.index_repo import RepoIndex, build_index
from src.ingest import fetch_repo
from src.llm import available_models, get_llm

app = FastAPI(title="CodeGraph", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_INDEXES: Dict[str, RepoIndex] = {}


class IngestReq(BaseModel):
    source: str  # GitHub URL or local path
    force: bool = False


class IngestResp(BaseModel):
    repo_id: str
    n_files: int
    n_symbols: int
    n_chunks: int


class AskReq(BaseModel):
    repo_id: str
    question: str
    model: str = "stub"
    session_id: str = "default"


class EvalReq(BaseModel):
    model: str = "stub"
    models: Optional[List[str]] = None
    eval_path: Optional[str] = None
    max_cases: Optional[int] = None
    repo_sources: Dict[str, str] = Field(default_factory=dict)
    run_ablation_study: bool = False

    @model_validator(mode="after")
    def validate_models(self) -> "EvalReq":
        if self.models is not None:
            deduped = list(dict.fromkeys(self.models))
            if not deduped:
                raise ValueError("models must contain at least one model id")
            self.models = deduped
        return self


@app.get("/health")
def health():
    return {"ok": True, "models": available_models(), "indexed": list(_INDEXES.keys())}


@app.post("/ingest", response_model=IngestResp)
def ingest(req: IngestReq):
    rid, root = fetch_repo(req.source, force=req.force)
    idx = build_index(rid, root)
    _INDEXES[rid] = idx
    return IngestResp(
        repo_id=rid,
        n_files=len(idx.parsed),
        n_symbols=sum(len(p.symbols) for p in idx.parsed.values()),
        n_chunks=len(idx.chunks),
    )


@app.post("/ingest/zip", response_model=IngestResp)
async def ingest_zip(file: UploadFile = File(...), force: bool = Form(False)):
    tmp = Path(tempfile.mkdtemp()) / (file.filename or "upload.zip")
    with tmp.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    rid, root = fetch_repo(str(tmp), force=force)
    idx = build_index(rid, root)
    _INDEXES[rid] = idx
    return IngestResp(
        repo_id=rid,
        n_files=len(idx.parsed),
        n_symbols=sum(len(p.symbols) for p in idx.parsed.values()),
        n_chunks=len(idx.chunks),
    )


@app.post("/ask")
def ask(req: AskReq):
    idx = _INDEXES.get(req.repo_id)
    if idx is None:
        raise HTTPException(404, f"Unknown repo_id {req.repo_id}; ingest first.")
    memory = SESSION_STORE.get(req.session_id)
    agent = CodeGraphAgent(
        idx, get_llm(req.model),
        memory=memory, session_id=req.session_id,
    )
    return agent.ask(req.question).to_dict()


@app.get("/file")
def file_view(repo_id: str, path: str, start: int = 1, end: int = 200):
    idx = _INDEXES.get(repo_id)
    if idx is None:
        raise HTTPException(404, "Unknown repo_id")
    return {"filepath": path, "start": start, "end": end, "text": idx.read_lines(path, start, end)}


@app.get("/graph/{repo_id}/{name}")
def graph_neighbors(repo_id: str, name: str, radius: int = 1):
    idx = _INDEXES.get(repo_id)
    if idx is None:
        raise HTTPException(404, "Unknown repo_id")
    sub = idx.graph.neighbors_subgraph(name, radius=radius)
    return {
        "nodes": [{"id": n, **sub.nodes[n]} for n in sub.nodes],
        "edges": [{"source": u, "target": v, **sub.edges[u, v]} for u, v in sub.edges],
    }


@app.post("/evaluate")
def evaluate(req: EvalReq):
    path = Path(req.eval_path) if req.eval_path else (DATA_DIR / "CodeGraphEval_50.json")
    if req.run_ablation_study:
        return run_ablation(
            path,
            model=req.model,
            repo_sources=req.repo_sources,
            max_cases=req.max_cases,
        )
    if req.models:
        result = run_eval_multi(
            path,
            models=req.models,
            repo_sources=req.repo_sources,
            max_cases=req.max_cases,
        )
        return format_multi_eval_results(result, models=req.models)
    return run_eval(path, model=req.model, repo_sources=req.repo_sources, max_cases=req.max_cases)
