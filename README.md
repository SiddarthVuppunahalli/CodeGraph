# CodeGraph: An Evaluation-Driven Agent for Codebase Understanding

**Proposed Application Area:** Option2 - LLMs + AI Agent System (Evaluation-First)

CodeGraph is a developer-focused assistant that answers repository questions with verifiable evidence. Every answer includes file-path and line-range citations, and the agent can follow imports/calls to explain behavior that spans multiple files.

## Team Members
*   **Siddarth Vuppunahalli** — Siddarth.vuppunahalli@sjsu.edu — SJSU ID: 019157203
*   **Krishna Panjiyar** — krishna.panjiyar@sjsu.edu — SJSU ID: 014981369
*   **Shivani Vinodkumar Jariwala** — shivanivinodkumar.jariwala@sjsu.edu — SJSU ID: 018284188

## Team ID
DL Group 12

## Dataset: CodeGraphEval-50
Because this is an evaluation-first agent system, our dataset is a curated benchmark:
1.  **Few-shot prompt examples (10–20 cases):** Demonstrations used for prompting/tuning.
2.  **Evaluation set (≥50 test cases):** `CodeGraphEval-50` built from public repositories. Each test case includes:
    *   `repo_name` and version/commit
    *   `question` (NL query)
    *   `category` (lookup / config-default / call-trace / dependency-impact)
    *   `ground_truth_answer`
    *   `ground_truth_evidence` (file paths + exact line ranges)
    *   `difficulty` (easy/medium/hard)

## Final Demo Vision
A web app where users provide a GitHub URL/ZIP and ask questions. CodeGraph will output:
1.  Answers with clickable file + line range citations.
2.  Optional import/call-path traces and dependency graphs.
3.  A tool-trace panel logging the Planner → tool calls → Critic checks.
4.  An "Evaluate" button that runs the CodeGraphEval-50 suite across LLMs (Llama-family, Qwen/DeepSeek coder, Hosted Model) to report accuracy, citation grounding score, and hallucination rate.

## Technologies Used
*   **Tree-sitter:** Code parsing
*   **FAISS:** Vector index
*   **rank-bm25:** Lexical retrieval
*   **NetworkX:** Graph construction/traversal (+ GraphRAG retrieval)
*   **Hugging Face Transformers:** Model loading/inference
*   **FastAPI & Streamlit/React:** Backend and UI
*   **Open-weight / Hosted LLMs**

## Quick Start (macOS with Virtual Environment)

### Prerequisites

Make sure you have Python 3.10+ and Xcode command line tools installed:

```bash
# Install Xcode command line tools (needed for tree-sitter compilation)
xcode-select --install

# Check Python version (macOS ships with python3 via Xcode or Homebrew)
python3 --version

# If python3 is not found, install via Homebrew
brew install python
```

### 1) Create and activate the virtual environment

```bash
cd CodeGraph-main
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the beginning of your terminal prompt. Every command below assumes the venv is active. If you open a new terminal tab, run `source .venv/bin/activate` again.

### 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Optional extras for real embeddings and hosted LLMs:

```bash
# For Apple Silicon (M1/M2/M3/M4) — CPU-only torch is lighter and sufficient
pip install sentence-transformers openai anthropic
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install accelerate
```

### 3) Set API keys (only needed for hosted models)

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

You can add these to your `~/.zshrc` or `~/.bash_profile` to persist across sessions.

### 4) Run the smoke tests (no API keys needed)

```bash
python -m pytest tests/test_smoke.py -x -v
```

This uses the `StubLLM` — a deterministic offline baseline — and runs 8 tests covering parsing, the agent loop, GraphRAG, session memory, grounding verification, structured logging, and stratified metrics.

### 5) Index a repo from the CLI

```bash
python -m scripts.ingest_repo https://github.com/psf/requests
```

### 6) Run the eval harness

```bash
# Single model (stub, no keys needed)
python -m scripts.run_eval --model stub --eval data/CodeGraphEval_50_sample.json

# Multi-model comparison table
python -m scripts.run_eval --models stub openai:gpt-4o-mini \
    anthropic:claude-haiku-4-5-20251001 hf:Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --eval data/CodeGraphEval_50_sample.json --out results.json
```

The eval output now includes `by_difficulty` and `by_category` breakdowns along with `avg_grounding_score` and `hallucination_rate` per model.

### 7) Run the web app

Open **two separate terminal tabs**. Activate the venv in both:

**Tab 1 — FastAPI backend:**
```bash
cd CodeGraph-main
source .venv/bin/activate
uvicorn src.api.main:app --reload
```

**Tab 2 — Streamlit UI:**
```bash
cd CodeGraph-main
source .venv/bin/activate
streamlit run src/ui/streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

**Troubleshooting the web app:**

If `localhost:8501` refuses to connect:
*   Try `http://127.0.0.1:8501` instead.
*   Make sure Tab 1 shows `Uvicorn running on http://127.0.0.1:8000` with no errors.
*   Make sure Tab 2 shows `You can now view your Streamlit app in your browser`.
*   If Streamlit is not found: `pip install streamlit` (it may not be in `requirements.txt`).
*   If the port is in use: `lsof -i :8501` to find the process, then `kill -9 <PID>`.
*   Try binding explicitly: `streamlit run src/ui/streamlit_app.py --server.address localhost --server.port 8501`

### 8) Deactivate when done

```bash
deactivate
```

## Key Features

### GraphRAG Retrieval
The agent includes a `graph_search` tool that performs graph-aware retrieval: it resolves seed function/class names in the dependency graph, then walks N hops outward to gather structural context (callers, callees, containing classes). This goes beyond basic vector search by using the code's actual dependency structure.

### Stateful Multi-Turn Memory
Each session maintains a memory store (`SessionMemory`) of prior Q&A exchanges. When you ask a follow-up question like "what calls that function?", the agent retrieves relevant prior context and chains it into the current prompt. Memory entries are summarized and keyword-indexed, not raw chat history. The `/ask` endpoint accepts a `session_id` parameter to isolate conversations.

### Grounding Verification (Critic)
The citation critic does two-stage validation:
1.  **Overlap check** — drops any citations that don't overlap evidence spans the tools actually returned.
2.  **Content verification** — re-fetches the cited source lines and checks whether salient terms from the answer actually appear in the code at those locations. Returns a `grounding_score` (0.0–1.0) per answer.

### Structured Logging
Every agent interaction emits JSON log entries to `logs/agent_log.jsonl`. Each entry includes: `timestamp`, `session_id`, `stage` (planner / tool_call / tool_result / critic / final_answer / error), `tool_name`, `tool_args`, `latency_ms`, `model`, `tokens_in`, `tokens_out`, and `grounding_score`.

### Difficulty-Stratified Evaluation
The eval harness reports metrics broken down by `difficulty` (easy/medium/hard) and `category` (call-trace, config-default, lookup, dependency-impact). This shows exactly where each model succeeds or fails rather than just an overall number.

## Layout

```
src/
  ingest/       fetch + walk repos (GitHub URL, zip, local dir)
  parsing/      tree-sitter Python parser (regex fallback)
  graph/        NetworkX call/contains graph + GraphRAG search
  retrieval/    chunker, BM25, FAISS vector, hybrid RRF
  llm/          pluggable backends: stub | openai | anthropic | hf
  agent/        ReAct loop with tools + citation critic
    memory.py   stateful multi-turn session memory
    logging.py  structured JSON logging
  eval/         CodeGraphEval-50 harness + grounding metrics
  api/          FastAPI backend
  ui/           Streamlit UI
scripts/        run_eval.py, ingest_repo.py
tests/          smoke tests (8 tests covering all features)
data/           CodeGraphEval-50 dataset
logs/           structured agent logs (auto-created)
```

## Project Roadmap & Next Steps

### Phase 1: Foundation (Current)
- [x] Initial Repository Setup and README.
- [x] Establish required dependencies (`transformers`, `tree-sitter`, `networkx`, `faiss`, `fastapi`).
- [x] Define the `CodeGraphEval-50` Evaluation Benchmark dataset schema and initial mock data.

### Phase 2: Core Infrastructure (Code Parsing & Retrieval)
- [x] Implement `tree-sitter` parsers (`src/parsing`) to extract functions, classes, and calls from code.
- [x] Build the Dependency Graph using `NetworkX` (`src/graph`) to track caller-callee relationships.
- [x] Implement Lexical Search (`rank-bm25`) and Semantic Search (`FAISS`) for repository context retrieval.
- [x] GraphRAG retrieval — seed node resolution + N-hop graph traversal for structural context.

### Phase 3: Agent System & Evaluation Harness
- [x] Develop the Agent logic (`src/agent`) to answer NLP queries and output strict citation ranges.
- [x] Build the Evaluation script (`scripts/run_eval.py`) to systematically grade the Agent against `CodeGraphEval-50`.
- [x] Stateful multi-turn session memory for chaining follow-up questions.
- [x] Grounding verification — critic re-fetches cited lines and validates answer claims.
- [x] Structured JSON logging for every planner, tool, and critic interaction.
- [x] Difficulty-stratified and category-stratified evaluation reporting.
- [ ] Integrate local Open-weight LLMs (e.g., Llama, Qwen coder) and a Hosted API LLM for comparison.

### Phase 4: Full App & Demo UI
- [x] Develop a `FastAPI` backend to serve the Agent logic.
- [x] Build the `Streamlit` Web application allowing users to interactively query codebases.
- [ ] Add the complete "Evaluate" dashboard to compare Accuracy, Grounding Score, and Hallucination Rates across models.
- [ ] Implement UI components to display dependency sub-graphs and call-path traces visually.
- [ ] Add safety/guardrails layer for detecting and refusing sensitive data exfiltration queries.
