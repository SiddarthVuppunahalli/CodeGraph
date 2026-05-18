# CodeGraph: An Evaluation-Driven Agent for Codebase Understanding

**Proposed Application Area:** Option 2 - LLMs + AI Agent System (Evaluation-First)

CodeGraph is a developer-focused assistant that answers repository questions with verifiable evidence. Every answer includes file-path and line-range citations, and the agent can follow imports and calls to explain behavior that spans multiple files.

## Team Members
- **Siddarth Vuppunahalli** - siddarth.vuppunahalli@sjsu.edu - SJSU ID: 019157203
- **Krishna Panjiyar** - krishna.panjiyar@sjsu.edu - SJSU ID: 014981369
- **Shivani Vinodkumar Jariwala** - shivanivinodkumar.jariwala@sjsu.edu - SJSU ID: 018284188

## Team ID
DL Group 12

## Dataset: CodeGraphEval-50
CodeGraphEval-50 is a validated 50-case benchmark for this repository.

Each case includes:
- `repo_name` and `version`
- `question`
- `category` (`lookup`, `config-default`, `call-trace`, `dependency-impact`)
- `ground_truth_answer`
- `ground_truth_evidence` with exact file paths and line ranges
- `difficulty` (`easy`, `medium`, `hard`)

The benchmark file lives at `data/CodeGraphEval_50.json`. You can regenerate it from live code with:

```bash
python -m scripts.generate_eval_dataset
```

## Demo Experience
The app supports:
- grounded Q&A with citations
- tool-trace inspection
- graph neighborhood visualization
- single-model evaluation
- multi-model comparison in the `Evaluate` tab with tables, charts, and per-case drilldowns

## Technologies Used
- Tree-sitter
- FAISS
- rank-bm25
- NetworkX
- Hugging Face Transformers
- FastAPI
- Streamlit

## Quick Start

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
. .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Optional extras for hosted and open-weight models:

```bash
pip install sentence-transformers openai anthropic
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install accelerate
```

### 3) Set API keys if needed

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 4) Run tests

```bash
python -m pytest tests -x -v
```

This covers smoke behavior, structured evaluation responses, and benchmark validation.

### 5) Index a repo from the CLI

```bash
python -m scripts.ingest_repo https://github.com/psf/requests
```

### 6) Run the eval harness

```bash
# Single model
python -m scripts.run_eval --model stub --eval data/CodeGraphEval_50.json

# Multi-model comparison
python -m scripts.run_eval --models stub openai:gpt-4o-mini \
    anthropic:claude-haiku-4-5-20251001 hf:Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --eval data/CodeGraphEval_50.json --out results.json
```

The eval output includes aggregate metrics plus `by_difficulty` and `by_category` breakdowns.

### 6a) Run the ablation study

```bash
python -m scripts.run_ablation
```

This runs five real configurations through the full eval harness on all 50 cases and writes `ablation_output/ablation_results.json`:

| Variant         | What it isolates                          |
| --------------- | ----------------------------------------- |
| `baseline-stub` | Default settings — control                |
| `no-grounding`  | Grounding critic disabled (threshold=-1)  |
| `low-topk`      | `top_k = 2` — under-retrieval             |
| `high-topk`     | `top_k = 16` — over-retrieval             |
| `smart-stub`    | Improved rule-based "LLM" with multi-tool exploration |

Expected output (deterministic — `PYTHONHASHSEED=42` pinned in the script):

```
variant              n  ans_cont  cite_gnd  cite_f1  hallucin  gnd_score
baseline-stub       50     0.320     0.440    0.338     0.560      0.840
no-grounding        50     0.380     0.440    0.338     0.560      0.840
low-topk            50     0.320     0.440    0.338     0.560      0.840
high-topk           50     0.320     0.440    0.338     0.560      0.840
smart-stub          50     0.380     0.420    0.322     0.580      0.907
```

See the project report (Section 6.7) for analysis of these numbers.

### 7) Run the web app

Terminal 1:

```bash
uvicorn src.api.main:app --reload
```

Terminal 2:

```bash
streamlit run src/ui/streamlit_app.py
```

Then open `http://localhost:8501`.

## Key Features

### GraphRAG Retrieval
The agent includes a `graph_search` tool that resolves seed function or class names in the dependency graph, then walks outward to gather structural context.

### Stateful Multi-Turn Memory
Each session maintains a `SessionMemory` of prior Q&A exchanges so follow-up questions can reuse earlier context.

### Grounding Verification
The critic re-fetches cited source lines and checks whether salient answer terms appear in the cited code, producing a `grounding_score`.

### Structured Logging
Every agent interaction emits JSON log entries to `logs/agent_log.jsonl`.

### Multi-Model Evaluation Dashboard
The `Evaluate` tab supports one-model and multi-model runs, comparison charts, difficulty/category breakdowns, and per-case drilldowns.

## Project Layout

```text
src/
  agent/        agent loop, memory, guardrails, logging
  api/          FastAPI backend
  eval/         harness, metrics, dataset validation
  graph/        dependency graph + GraphRAG search
  ingest/       repo fetching and file walking
  llm/          pluggable model backends
  parsing/      tree-sitter parsing
  retrieval/    chunking and retrieval
  ui/           Streamlit UI
scripts/        CLI helpers and dataset generation
tests/          smoke and evaluation dashboard tests
data/           CodeGraphEval-50 benchmark
logs/           structured agent logs
```

## Roadmap Snapshot
- [x] Parsing, graph construction, lexical + semantic retrieval
- [x] Agent loop with citations and grounding verification
- [x] Structured evaluation harness
- [x] Multi-model evaluation dashboard in the web app
- [x] Validated 50-case benchmark for the current repository
- [ ] Broaden the benchmark to multiple external repositories
- [ ] Add richer graph/path visualizations
