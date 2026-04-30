"""End-to-end smoke test using the StubLLM and a tiny synthetic repo."""
from __future__ import annotations
from pathlib import Path

from src.agent import CodeGraphAgent
from src.eval.metrics import score_case
from src.index_repo import build_index
from src.llm import get_llm


SAMPLE_AUTH = '''\
def authenticate_user(username, password):
    """Verify a username/password pair."""
    return username == "admin" and password == "secret"


def login_handler(request):
    user = request.get("user")
    pw = request.get("pw")
    if authenticate_user(user, pw):
        return {"status": "ok"}
    return {"status": "denied"}
'''

SAMPLE_CONFIG = '''\
# Application configuration
DEBUG = False
TIMEOUT = 30
DATABASE_URL = "sqlite:///app.db"
'''


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / "auth.py").write_text(SAMPLE_AUTH)
    (tmp_path / "config.py").write_text(SAMPLE_CONFIG)
    return tmp_path


def test_parse_and_graph(tmp_path):
    root = _make_repo(tmp_path)
    idx = build_index("test", root)
    names = {s.name for pf in idx.parsed.values() for s in pf.symbols}
    assert {"authenticate_user", "login_handler"}.issubset(names)
    callers = idx.graph.get_callers("authenticate_user")
    assert any("login_handler" in q for q, _ in callers)


def test_agent_with_stub_llm(tmp_path):
    root = _make_repo(tmp_path)
    idx = build_index("test", root)
    agent = CodeGraphAgent(idx, get_llm("stub"))
    result = agent.ask("What calls the authenticate_user function?")
    assert result.citations, f"expected citations, got: {result.to_dict()}"
    assert any(c.filepath.endswith("auth.py") for c in result.citations)
    assert result.grounded


def test_metrics_grounding(tmp_path):
    case = {
        "question": "Where is the timeout?",
        "ground_truth_answer": "TIMEOUT is 30 in config.py",
        "category": "config-default", "difficulty": "easy",
        "ground_truth_evidence": [{"filepath": "config.py", "line_ranges": [3, 3]}],
    }
    agent_out = {
        "answer": "TIMEOUT is 30 in config.py",
        "citations": [{"filepath": "config.py", "line_ranges": [1, 5]}],
        "latency_s": 0.01,
    }
    m = score_case(case, agent_out)
    assert m.citation_grounded == 1
    assert m.hallucinated == 0
    assert m.citation_f1 > 0
