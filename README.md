# CodeGraph: An Evaluation-Driven Agent for Codebase Understanding

**Proposed Application Area:** Option2 - LLMs + AI Agent System (Evaluation-First)

CodeGraph is a developer-focused assistant that answers repository questions with verifiable evidence. Every answer includes file-path and line-range citations, and the agent can follow imports/calls to explain behavior that spans multiple files.

## Team Members
*   **Siddarth Vuppunahalli** — Siddarth.vuppunahalli@sjsu.edu — SJSU ID: 019157203
*   **Krishna Panjiyar** — krishna.panjiyar@sjsu.edu — SJSU ID: 014981369
*   **Shivani Vinodkumar Jariwala** — shivanivinodkumar.jariwala@sjsu.edu — SJSU ID: 018284188

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
*   **NetworkX:** Graph construction/traversal
*   **Hugging Face Transformers:** Model loading/inference
*   **FastAPI & Streamlit/React:** Backend and UI
*   **Open-weight / Hosted LLMs**

## Project Roadmap & Next Steps

### Phase 1: Foundation (Current)
- [x] Initial Repository Setup and README.
- [x] Establish required dependencies (`transformers`, `tree-sitter`, `networkx`, `faiss`, `fastapi`).
- [x] Define the `CodeGraphEval-50` Evaluation Benchmark dataset schema and initial mock data.

### Phase 2: Core Infrastructure (Code Parsing & Retrieval)
- [ ] Implement `tree-sitter` parsers (`src/parsing`) to extract functions, classes, and calls from code. (*Stubs created*)
- [ ] Build the Dependency Graph using `NetworkX` (`src/graph`) to track caller-callee relationships. (*Basic framework created*)
- [ ] Implement Lexical Search (`rank-bm25`) and Semantic Search (`FAISS`) for repository context retrieval.

### Phase 3: Agent System & Evaluation Harness
- [ ] Develop the Agent logic (`src/agent`) to answer NLP queries and output strict citation ranges.
- [ ] Build the Evaluation script (`scripts/run_eval.py`) to systematically grade the Agent against `CodeGraphEval-50`. (*Initial harness created*)
- [ ] Integrate local Open-weight LLMs (e.g., Llama, Qwen coder) and a Hosted API LLM for comparison.

### Phase 4: Full App & Demo UI
- [ ] Develop a `FastAPI` backend to serve the Agent logic.
- [ ] Build the `Streamlit` or `React` Web application allowing users to interactively query codebases.
- [ ] Add the complete "Evaluate" dashboard to compare Accuracy, Grounding Score, and Hallucination Rates across models.
- [ ] Implement UI components to display dependency sub-graphs and call-path traces visually.
