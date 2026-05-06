"""End-to-end smoke test using the StubLLM and a tiny synthetic repo."""
from __future__ import annotations
from pathlib import Path

from src.agent import CodeGraphAgent, SessionMemory
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
        "grounding_score": 1.0,
    }
    m = score_case(case, agent_out)
    assert m.citation_grounded == 1
    assert m.hallucinated == 0
    assert m.citation_f1 > 0
    assert m.grounding_score == 1.0


# ---------- NEW: Feature tests ----------

def test_graph_search_tool(tmp_path):
    """GraphRAG: graph_search finds neighbors of a seed node."""
    root = _make_repo(tmp_path)
    idx = build_index("test", root)
    results = idx.graph.graph_search(["authenticate_user"], hops=1)
    qnames = [qn for qn, _ in results]
    # Should find login_handler as a caller within 1 hop
    assert any("login_handler" in qn for qn in qnames), f"got: {qnames}"


def test_session_memory_multi_turn(tmp_path):
    """Stateful memory: second question can chain on prior answer."""
    root = _make_repo(tmp_path)
    idx = build_index("test", root)
    memory = SessionMemory()
    agent = CodeGraphAgent(idx, get_llm("stub"), memory=memory)

    # First question
    r1 = agent.ask("What calls the authenticate_user function?")
    assert r1.grounded
    assert len(memory.entries) == 1

    # Second question — memory should include prior context
    r2 = agent.ask("Where is the default timeout configuration defined?")
    assert len(memory.entries) == 2
    # Memory should have symbols from both answers
    all_syms = memory.get_all_symbols()
    assert len(all_syms) > 0


def test_grounding_score(tmp_path):
    """Grounding verification: score reflects whether citations match answer."""
    root = _make_repo(tmp_path)
    idx = build_index("test", root)
    agent = CodeGraphAgent(idx, get_llm("stub"))
    result = agent.ask("What calls the authenticate_user function?")
    # The stub answer mentions symbols that exist in auth.py,
    # so grounding_score should be > 0
    assert result.grounding_score > 0.0, f"grounding_score={result.grounding_score}"
    result_dict = result.to_dict()
    assert "grounding_score" in result_dict


def test_structured_logging(tmp_path):
    """Structured logging: agent emits log entries to file."""
    root = _make_repo(tmp_path)
    idx = build_index("test", root)

    # Clear the log file before this test
    from src.config import ROOT
    log_file = ROOT / "logs" / "agent_log.jsonl"
    log_file.write_text("")

    agent = CodeGraphAgent(idx, get_llm("stub"), session_id="test-session")
    agent.ask("What calls the authenticate_user function?")

    assert log_file.exists(), "log file not created"
    import json
    lines = log_file.read_text().strip().split("\n")
    entries = [json.loads(line) for line in lines if line.strip()]
    # Should have planner, tool_call, tool_result, critic, final_answer entries
    stages = {e["stage"] for e in entries}
    assert "planner" in stages, f"missing planner, got: {stages}"
    assert "final_answer" in stages, f"missing final_answer, got: {stages}"
    # Verify session_id is tracked
    assert all(e["session_id"] == "test-session" for e in entries)


def test_difficulty_stratified_metrics(tmp_path):
    """Eval: aggregate includes by_difficulty and by_category breakdowns."""
    from src.eval.metrics import aggregate, CaseMetrics
    cases = [
        CaseMetrics(case_id="q1", category="call-trace", difficulty="easy",
                    answer_em=1, answer_contains=1, citation_grounded=1,
                    citation_precision=1.0, citation_recall=1.0, citation_f1=1.0,
                    hallucinated=0, grounding_score=1.0, latency_s=0.1),
        CaseMetrics(case_id="q2", category="config-default", difficulty="hard",
                    answer_em=0, answer_contains=1, citation_grounded=0,
                    citation_precision=0.0, citation_recall=0.0, citation_f1=0.0,
                    hallucinated=1, grounding_score=0.0, latency_s=0.2),
    ]
    agg = aggregate(cases)
    assert "by_difficulty" in agg
    assert "easy" in agg["by_difficulty"]
    assert "hard" in agg["by_difficulty"]
    assert agg["by_difficulty"]["easy"]["answer_em"] == 1.0
    assert agg["by_difficulty"]["hard"]["hallucination_rate"] == 1.0
    assert "by_category" in agg
    assert "call-trace" in agg["by_category"]
    assert "avg_grounding_score" in agg
