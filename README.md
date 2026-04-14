# CodeGraph: An Evaluation-Driven Agent for Codebase Understanding

**Proposed Application Area:** Option2 - LLMs + AI Agent System (Evaluation-First)

CodeGraph is a developer-focused assistant that answers repository questions with verifiable evidence. Every answer includes file-path and line-range citations, and the agent can follow imports/calls to explain behavior that spans multiple files.

## Team Members
*   **Siddarth Vuppunhalli** — Siddarth.vuppunhalli@sjsu.edu — SJSU ID: 019157203
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

## Progress & Next Steps
- [x] Initial Repository Setup and README.
- [ ] Define CodeGraphEval-50 JSON Schema.
- [ ] Implement initial NetworkX codebase graph builder.
- [ ] Set up evaluation harness script to test logic against dummy data.
- [ ] Implement basic Retrieval (FAISS/BM25).
