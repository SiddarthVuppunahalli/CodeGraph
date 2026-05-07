from __future__ import annotations

import json
from pathlib import Path

from src.config import ROOT
from src.index_repo import build_index

VERSION = "v1.0.0"


def _build_symbol_map() -> dict[tuple[str, str, str], dict]:
    idx = build_index("CodeGraph", ROOT)
    symbols: dict[tuple[str, str, str], dict] = {}
    for filepath, parsed in idx.parsed.items():
        if not filepath.startswith("src/") and not filepath.startswith("scripts/") and not filepath.startswith("tests/"):
            continue
        for symbol in parsed.symbols:
            symbols[(filepath, symbol.kind, symbol.name)] = {
                "filepath": filepath,
                "kind": symbol.kind,
                "name": symbol.name,
                "start_line": symbol.start_line,
                "end_line": symbol.end_line,
                "qualified_name": symbol.qualified_name,
            }
    return symbols


def _symbol(symbols: dict, filepath: str, kind: str, name: str) -> dict:
    key = (filepath, kind, name)
    if key not in symbols:
        raise KeyError(f"Missing symbol {key}")
    return symbols[key]


def _line_span(filepath: str, needle: str) -> dict:
    target = ROOT / filepath
    for line_no, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        if needle in line:
            return {"filepath": filepath, "line_ranges": [line_no, line_no]}
    raise ValueError(f"Could not find '{needle}' in {filepath}")


def _rhs(filepath: str, field_name: str) -> str:
    target = ROOT / filepath
    for line in target.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{field_name}:") and "=" in stripped:
            return stripped.split("=", 1)[1].strip()
    raise ValueError(f"Could not extract default for {field_name} from {filepath}")


def _case(question: str, category: str, answer: str, evidence: list[dict], difficulty: str, notes: str = "") -> dict:
    return {
        "repo_name": "CodeGraph",
        "version": VERSION,
        "question": question,
        "category": category,
        "ground_truth_answer": answer,
        "ground_truth_evidence": evidence,
        "difficulty": difficulty,
        "notes": notes,
    }


def _lookup_case(symbols: dict, filepath: str, kind: str, name: str, difficulty: str) -> dict:
    symbol = _symbol(symbols, filepath, kind, name)
    noun = {"class": "class", "function": "function", "method": "method"}[kind]
    return _case(
        question=f"Where is {name} defined?",
        category="lookup",
        answer=f"The {noun} {name} is defined in {filepath}.",
        evidence=[{"filepath": filepath, "line_ranges": [symbol["start_line"], symbol["end_line"]]}],
        difficulty=difficulty,
        notes=f"kind={kind}",
    )


def _config_case(filepath: str, field_name: str, owner: str, difficulty: str) -> dict:
    rhs = _rhs(filepath, field_name)
    return _case(
        question=f"What is the default value of {field_name} in {owner}?",
        category="config-default",
        answer=f"The default value of {field_name} in {owner} is {rhs}.",
        evidence=[_line_span(filepath, f"{field_name}:")],
        difficulty=difficulty,
        notes=f"default={rhs}",
    )


def generate() -> list[dict]:
    symbols = _build_symbol_map()
    cases: list[dict] = []

    # Lookup cases (14)
    cases.extend([
        _lookup_case(symbols, "src/agent/agent.py", "class", "CodeGraphAgent", "easy"),
        _lookup_case(symbols, "src/api/main.py", "class", "EvalReq", "easy"),
        _lookup_case(symbols, "src/eval/harness.py", "function", "format_multi_eval_results", "medium"),
        _lookup_case(symbols, "src/eval/dataset.py", "function", "validate_eval_dataset", "medium"),
        _lookup_case(symbols, "src/graph/builder.py", "class", "DependencyGraphBuilder", "easy"),
        _lookup_case(symbols, "src/index_repo.py", "function", "build_index", "easy"),
        _lookup_case(symbols, "src/ingest/fetch.py", "function", "fetch_repo", "easy"),
        _lookup_case(symbols, "src/llm/registry.py", "function", "available_models", "easy"),
        _lookup_case(symbols, "src/llm/registry.py", "function", "get_llm", "easy"),
        _lookup_case(symbols, "src/parsing/parser.py", "class", "PythonCodeParser", "easy"),
        _lookup_case(symbols, "src/retrieval/hybrid.py", "class", "HybridRetriever", "easy"),
        _lookup_case(symbols, "src/retrieval/vector_index.py", "class", "VectorIndex", "easy"),
        _lookup_case(symbols, "src/agent/tools.py", "class", "ToolBox", "medium"),
        _lookup_case(symbols, "src/agent/guardrails.py", "function", "scan_answer", "medium"),
    ])

    # Config/default cases (10)
    cases.extend([
        _config_case("src/config.py", "embed_model", "Settings", "easy"),
        _config_case("src/config.py", "embed_dim", "Settings", "easy"),
        _config_case("src/config.py", "top_k", "Settings", "easy"),
        _config_case("src/config.py", "max_agent_steps", "Settings", "easy"),
        _config_case("src/config.py", "default_model", "Settings", "easy"),
        _config_case("src/config.py", "grounding_threshold", "Settings", "medium"),
        _config_case("src/config.py", "max_retries", "Settings", "medium"),
        _config_case("src/agent/agent.py", "grounded", "AgentResult", "medium"),
        _config_case("src/agent/agent.py", "grounding_score", "AgentResult", "medium"),
        _config_case("src/agent/agent.py", "model", "AgentResult", "medium"),
    ])

    # Call-trace cases (13)
    cases.extend([
        _case(
            "What calls check_query?",
            "call-trace",
            "check_query is called by CodeGraphAgent.ask.",
            [{"filepath": "src/agent/agent.py", "line_ranges": [80, 242]}],
            "medium",
            "n_callers=1",
        ),
        _case(
            "What calls scan_answer?",
            "call-trace",
            "scan_answer is called by CodeGraphAgent.ask and test_answer_redaction.",
            [
                {"filepath": "src/agent/agent.py", "line_ranges": [80, 242]},
                {"filepath": "tests/test_smoke.py", "line_ranges": [198, 212]},
            ],
            "hard",
            "n_callers=2",
        ),
        _case(
            "What calls graph_search?",
            "call-trace",
            "graph_search is called by ToolBox._graph_search and test_graph_search_tool.",
            [
                {"filepath": "src/agent/tools.py", "line_ranges": [91, 114]},
                {"filepath": "tests/test_smoke.py", "line_ranges": [80, 87]},
            ],
            "hard",
            "n_callers=2",
        ),
        _case(
            "What calls get_callers?",
            "call-trace",
            "get_callers is called by ToolBox._callers and test_parse_and_graph.",
            [
                {"filepath": "src/agent/tools.py", "line_ranges": [60, 71]},
                {"filepath": "tests/test_smoke.py", "line_ranges": [39, 45]},
            ],
            "hard",
            "n_callers=2",
        ),
        _case(
            "What calls score_case?",
            "call-trace",
            "score_case is called by run_eval and test_metrics_grounding.",
            [
                {"filepath": "src/eval/harness.py", "line_ranges": [59, 96]},
                {"filepath": "tests/test_smoke.py", "line_ranges": [58, 75]},
            ],
            "hard",
            "n_callers=2",
        ),
        _case(
            "What calls aggregate?",
            "call-trace",
            "aggregate is called by run_eval and test_difficulty_stratified_metrics.",
            [
                {"filepath": "src/eval/harness.py", "line_ranges": [59, 96]},
                {"filepath": "tests/test_smoke.py", "line_ranges": [148, 169]},
            ],
            "hard",
            "n_callers=2",
        ),
        _case(
            "What calls SessionMemory?",
            "call-trace",
            "SessionMemory is called by SessionStore.get and test_session_memory_multi_turn.",
            [
                {"filepath": "src/agent/memory.py", "line_ranges": [101, 104]},
                {"filepath": "tests/test_smoke.py", "line_ranges": [90, 107]},
            ],
            "hard",
            "n_callers=2",
        ),
        _case(
            "What calls StructuredLogger?",
            "call-trace",
            "StructuredLogger is called by CodeGraphAgent.ask and test_re_retrieval_logger_method.",
            [
                {"filepath": "src/agent/agent.py", "line_ranges": [80, 242]},
                {"filepath": "tests/test_smoke.py", "line_ranges": [236, 243]},
            ],
            "hard",
            "n_callers=2",
        ),
        _case(
            "What calls read_lines?",
            "call-trace",
            "read_lines is called by CodeGraphAgent._verify_grounding, ToolBox._read, and file_view.",
            [
                {"filepath": "src/agent/agent.py", "line_ranges": [260, 289]},
                {"filepath": "src/agent/tools.py", "line_ranges": [86, 89]},
                {"filepath": "src/api/main.py", "line_ranges": [110, 114]},
            ],
            "hard",
            "n_callers=3",
        ),
        _case(
            "What calls run_eval_multi?",
            "call-trace",
            "run_eval_multi is called by scripts.run_eval.main and src.api.main.evaluate.",
            [
                {"filepath": "scripts/run_eval.py", "line_ranges": [10, 44]},
                {"filepath": "src/api/main.py", "line_ranges": [129, 140]},
            ],
            "medium",
            "n_callers=2",
        ),
        _case(
            "What calls walk_python_files?",
            "call-trace",
            "walk_python_files is called by build_index.",
            [{"filepath": "src/index_repo.py", "line_ranges": [37, 66]}],
            "medium",
            "n_callers=1",
        ),
        _case(
            "What calls chunk_parsed_file?",
            "call-trace",
            "chunk_parsed_file is called by build_index.",
            [{"filepath": "src/index_repo.py", "line_ranges": [37, 66]}],
            "medium",
            "n_callers=1",
        ),
        _case(
            "What calls format_multi_eval_results?",
            "call-trace",
            "format_multi_eval_results is called by evaluate.",
            [{"filepath": "src/api/main.py", "line_ranges": [129, 140]}],
            "medium",
            "n_callers=1",
        ),
    ])

    # Dependency-impact cases (13)
    cases.extend([
        _case(
            "If run_eval changes, which entrypoints are directly affected?",
            "dependency-impact",
            "The directly affected entrypoints are scripts.run_eval.main, src.api.main.evaluate, and run_eval_multi.",
            [
                {"filepath": "scripts/run_eval.py", "line_ranges": [10, 44]},
                {"filepath": "src/api/main.py", "line_ranges": [129, 140]},
                {"filepath": "src/eval/harness.py", "line_ranges": [99, 100]},
            ],
            "hard",
            "depends_on=run_eval",
        ),
        _case(
            "If build_index changes, which entrypoints are directly affected?",
            "dependency-impact",
            "The directly affected entrypoints are scripts.ingest_repo.main, ingest, ingest_zip, and run_eval.",
            [
                {"filepath": "scripts/ingest_repo.py", "line_ranges": [9, 20]},
                {"filepath": "src/api/main.py", "line_ranges": [68, 77]},
                {"filepath": "src/api/main.py", "line_ranges": [80, 93]},
                {"filepath": "src/eval/harness.py", "line_ranges": [59, 96]},
            ],
            "hard",
            "depends_on=build_index",
        ),
        _case(
            "If fetch_repo changes, which entrypoints are directly affected?",
            "dependency-impact",
            "The directly affected entrypoints are scripts.ingest_repo.main, ingest, ingest_zip, and run_eval.",
            [
                {"filepath": "scripts/ingest_repo.py", "line_ranges": [9, 20]},
                {"filepath": "src/api/main.py", "line_ranges": [68, 77]},
                {"filepath": "src/api/main.py", "line_ranges": [80, 93]},
                {"filepath": "src/eval/harness.py", "line_ranges": [59, 96]},
            ],
            "hard",
            "depends_on=fetch_repo",
        ),
        _case(
            "If read_lines changes, which code paths are directly affected?",
            "dependency-impact",
            "The directly affected code paths are CodeGraphAgent._verify_grounding, ToolBox._read, and file_view.",
            [
                {"filepath": "src/agent/agent.py", "line_ranges": [260, 289]},
                {"filepath": "src/agent/tools.py", "line_ranges": [86, 89]},
                {"filepath": "src/api/main.py", "line_ranges": [110, 114]},
            ],
            "hard",
            "depends_on=read_lines",
        ),
        _case(
            "Which function constructs the CodeGraphAgent used during evaluation?",
            "dependency-impact",
            "run_eval constructs the CodeGraphAgent used during evaluation.",
            [{"filepath": "src/eval/harness.py", "line_ranges": [59, 96]}],
            "medium",
            "constructs=CodeGraphAgent",
        ),
        _case(
            "Which function constructs the DependencyGraphBuilder during indexing?",
            "dependency-impact",
            "build_index constructs the DependencyGraphBuilder during indexing.",
            [{"filepath": "src/index_repo.py", "line_ranges": [37, 66]}],
            "medium",
            "constructs=DependencyGraphBuilder",
        ),
        _case(
            "Which function constructs the HybridRetriever returned inside RepoIndex?",
            "dependency-impact",
            "build_index constructs the HybridRetriever returned inside RepoIndex.",
            [{"filepath": "src/index_repo.py", "line_ranges": [37, 66]}],
            "medium",
            "constructs=HybridRetriever",
        ),
        _case(
            "Which method depends on scan_answer to sanitize final answers?",
            "dependency-impact",
            "CodeGraphAgent.ask depends on scan_answer to sanitize final answers.",
            [{"filepath": "src/agent/agent.py", "line_ranges": [80, 242]}],
            "medium",
            "depends_on=scan_answer",
        ),
        _case(
            "Which method depends on check_query to block sensitive prompts before tool use?",
            "dependency-impact",
            "CodeGraphAgent.ask depends on check_query to block sensitive prompts before tool use.",
            [{"filepath": "src/agent/agent.py", "line_ranges": [80, 242]}],
            "medium",
            "depends_on=check_query",
        ),
        _case(
            "Which method depends on graph_search to expose graph-aware retrieval through the toolbox?",
            "dependency-impact",
            "ToolBox._graph_search depends on graph_search to expose graph-aware retrieval through the toolbox.",
            [{"filepath": "src/agent/tools.py", "line_ranges": [91, 114]}],
            "medium",
            "depends_on=graph_search",
        ),
        _case(
            "Which function depends on walk_python_files to enumerate Python files before parsing?",
            "dependency-impact",
            "build_index depends on walk_python_files to enumerate Python files before parsing.",
            [{"filepath": "src/index_repo.py", "line_ranges": [37, 66]}],
            "medium",
            "depends_on=walk_python_files",
        ),
        _case(
            "Which function depends on chunk_parsed_file to create retrieval chunks?",
            "dependency-impact",
            "build_index depends on chunk_parsed_file to create retrieval chunks.",
            [{"filepath": "src/index_repo.py", "line_ranges": [37, 66]}],
            "medium",
            "depends_on=chunk_parsed_file",
        ),
        _case(
            "Which API endpoint depends on format_multi_eval_results to shape the comparison response?",
            "dependency-impact",
            "evaluate depends on format_multi_eval_results to shape the comparison response.",
            [{"filepath": "src/api/main.py", "line_ranges": [129, 140]}],
            "medium",
            "depends_on=format_multi_eval_results",
        ),
    ])

    if len(cases) != 50:
        raise ValueError(f"Expected 50 cases, found {len(cases)}")
    return cases


def main() -> None:
    out_path = ROOT / "data" / "CodeGraphEval_50.json"
    out_path.write_text(json.dumps(generate(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
