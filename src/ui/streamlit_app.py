"""CodeGraph Streamlit UI. Talks to the FastAPI backend.

Run:
    uvicorn src.api.main:app --reload         # in one terminal
    streamlit run src/ui/streamlit_app.py     # in another
"""
from __future__ import annotations
import json
import os

import requests
import streamlit as st

API = os.environ.get("CODEGRAPH_API", "http://localhost:8000")
st.set_page_config(page_title="CodeGraph", layout="wide")
st.title("CodeGraph — grounded Q&A over code")

if "repo_id" not in st.session_state:
    st.session_state.repo_id = None
if "stats" not in st.session_state:
    st.session_state.stats = None

with st.sidebar:
    st.header("1) Ingest a repo")
    src = st.text_input("GitHub URL or local path", placeholder="https://github.com/psf/requests")
    if st.button("Ingest", use_container_width=True) and src:
        with st.spinner("Cloning + indexing..."):
            r = requests.post(f"{API}/ingest", json={"source": src}, timeout=600)
        if r.ok:
            d = r.json()
            st.session_state.repo_id = d["repo_id"]
            st.session_state.stats = d
            st.success(f"Indexed {d['n_files']} files / {d['n_symbols']} symbols / {d['n_chunks']} chunks")
        else:
            st.error(r.text)

    up = st.file_uploader("...or upload a .zip", type=["zip"])
    if up and st.button("Ingest ZIP", use_container_width=True):
        with st.spinner("Indexing zip..."):
            r = requests.post(f"{API}/ingest/zip", files={"file": (up.name, up.getvalue())})
        if r.ok:
            d = r.json()
            st.session_state.repo_id = d["repo_id"]
            st.session_state.stats = d
            st.success(f"Indexed {d['n_files']} files")
        else:
            st.error(r.text)

    st.divider()
    st.header("Model")
    model = st.selectbox("LLM", [
        "stub", "openai:gpt-4o-mini", "anthropic:claude-haiku-4-5-20251001",
        "hf:Qwen/Qwen2.5-Coder-1.5B-Instruct",
    ])

    if st.session_state.stats:
        st.caption(f"repo_id: `{st.session_state.repo_id}`")
        st.json(st.session_state.stats)

tab_ask, tab_eval, tab_graph = st.tabs(["Ask", "Evaluate", "Graph"])

with tab_ask:
    q = st.text_input("Question", placeholder="Where is auth enforced?")
    if st.button("Ask", type="primary") and q and st.session_state.repo_id:
        with st.spinner("Thinking..."):
            r = requests.post(f"{API}/ask", json={
                "repo_id": st.session_state.repo_id, "question": q, "model": model,
            }, timeout=600)
        if not r.ok:
            st.error(r.text)
        else:
            d = r.json()
            colA, colB = st.columns([2, 1])
            with colA:
                st.subheader("Answer")
                st.write(d["answer"])
                st.caption(f"grounded={d['grounded']} · model={d['model']} · {d['latency_s']:.2f}s")
                st.subheader("Citations")
                for c in d["citations"]:
                    fp, (s, e) = c["filepath"], c["line_ranges"]
                    with st.expander(f"{fp}:{s}-{e}"):
                        fr = requests.get(f"{API}/file", params={
                            "repo_id": st.session_state.repo_id,
                            "path": fp, "start": s, "end": e,
                        })
                        if fr.ok:
                            st.code(fr.json()["text"], language="python")
            with colB:
                st.subheader("Tool trace")
                for step in d["trace"]:
                    with st.expander(f"step {step['step']}: {list(step['decision'].keys())[0]}"):
                        st.json(step["decision"])

with tab_eval:
    st.write("Run CodeGraphEval against the selected model and view metrics.")
    eval_path = st.text_input("Eval JSON", value="data/CodeGraphEval_50_sample.json")
    max_cases = st.number_input("Max cases (0 = all)", min_value=0, value=0)
    if st.button("Run evaluation"):
        body = {"model": model, "eval_path": eval_path}
        if max_cases:
            body["max_cases"] = int(max_cases)
        with st.spinner("Evaluating..."):
            r = requests.post(f"{API}/evaluate", json=body, timeout=3600)
        if not r.ok:
            st.error(r.text)
        else:
            d = r.json()
            st.subheader("Aggregate")
            st.json(d["aggregate"])
            st.subheader("Per-case")
            st.dataframe(d["cases"])

with tab_graph:
    name = st.text_input("Symbol (function/class) name to view neighborhood")
    radius = st.slider("Radius", 1, 3, 1)
    if st.button("Show graph") and name and st.session_state.repo_id:
        r = requests.get(f"{API}/graph/{st.session_state.repo_id}/{name}", params={"radius": radius})
        if r.ok:
            data = r.json()
            try:
                from streamlit_agraph import agraph, Node, Edge, Config
                nodes = [Node(id=n["id"], label=n.get("name", n["id"])) for n in data["nodes"]]
                edges = [Edge(source=e["source"], target=e["target"], label=e.get("type", "")) for e in data["edges"]]
                agraph(nodes=nodes, edges=edges, config=Config(width=900, height=600, directed=True))
            except Exception:
                st.json(data)
        else:
            st.error(r.text)
