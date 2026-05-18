"""CodeGraph Streamlit UI. Talks to the FastAPI backend.

Run:
    uvicorn src.api.main:app --reload         # in one terminal
    streamlit run src/ui/streamlit_app.py     # in another
"""
from __future__ import annotations

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API = os.environ.get("CODEGRAPH_API", "http://localhost:8000")
SUPPORTED_MODELS = [
    "stub",
    "openai:gpt-4o-mini",
    "anthropic:claude-haiku-4-5-20251001",
    "hf:Qwen/Qwen2.5-Coder-1.5B-Instruct",
]
SUMMARY_METRICS = [
    "answer_em",
    "answer_contains",
    "citation_grounded",
    "citation_f1",
    "hallucination_rate",
    "avg_grounding_score",
    "avg_latency_s",
]


def _handle_stale_repo(r: requests.Response) -> bool:
    if r.status_code != 404:
        return False
    text = r.text or ""
    if "Unknown repo_id" not in text:
        return False
    st.session_state.repo_id = None
    st.session_state.stats = None
    st.error("The backend was restarted and forgot the indexed repo. Please ingest the repo again.")
    return True


def _summary_frame(summary_rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(summary_rows)
    if frame.empty:
        return frame
    ordered_cols = ["model", "n", *SUMMARY_METRICS]
    return frame[[c for c in ordered_cols if c in frame.columns]]


def _breakdown_frame(aggregates_by_model: dict, key: str) -> pd.DataFrame:
    rows: list[dict] = []
    for model, aggregate in aggregates_by_model.items():
        for bucket, metrics in aggregate.get(key, {}).items():
            rows.append({"model": model, key[:-3]: bucket, **metrics})
    return pd.DataFrame(rows)


def _merge_case_rows(cases_by_model: dict[str, list[dict]]) -> pd.DataFrame:
    merged: dict[str, dict] = {}
    metric_cols = [
        "answer_em",
        "answer_contains",
        "citation_grounded",
        "citation_f1",
        "hallucinated",
        "grounding_score",
        "latency_s",
    ]
    for model, cases in cases_by_model.items():
        for case in cases:
            key = case.get("case_id", "")
            row = merged.setdefault(
                key,
                {
                    "case_id": key,
                    "category": case.get("category", ""),
                    "difficulty": case.get("difficulty", ""),
                },
            )
            for col in metric_cols:
                row[f"{model}::{col}"] = case.get(col)
    return pd.DataFrame(merged.values())


def _render_eval_overview(summary_df: pd.DataFrame) -> None:
    st.subheader("Comparison Table")
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("Leaderboard")
    lead_cols = st.columns(3)
    best_em = summary_df.sort_values("answer_em", ascending=False).iloc[0]
    best_grounding = summary_df.sort_values("avg_grounding_score", ascending=False).iloc[0]
    best_latency = summary_df.sort_values("avg_latency_s", ascending=True).iloc[0]
    lead_cols[0].metric("Best Exact Match", best_em["model"], f"{best_em['answer_em']:.2f}")
    lead_cols[1].metric("Best Grounding", best_grounding["model"], f"{best_grounding['avg_grounding_score']:.2f}")
    lead_cols[2].metric("Fastest Model", best_latency["model"], f"{best_latency['avg_latency_s']:.2f}s")

    quality_df = summary_df.melt(
        id_vars=["model"],
        value_vars=["answer_em", "answer_contains", "citation_f1"],
        var_name="metric",
        value_name="value",
    )
    quality_chart = alt.Chart(quality_df).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("value:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("model:N", title=None, sort="-x"),
        color=alt.Color("metric:N", title="Metric"),
        column=alt.Column("metric:N", title=None),
        tooltip=["model", "metric", alt.Tooltip("value:Q", format=".3f")],
    ).properties(height=220)
    st.subheader("Quality Comparison")
    st.caption("Higher is better for exact match, answer coverage, and citation F1.")
    st.altair_chart(quality_chart, use_container_width=True)

    risk_df = summary_df.copy()
    risk_df["trust_score"] = 1.0 - risk_df["hallucination_rate"]
    risk_melt = risk_df.melt(
        id_vars=["model"],
        value_vars=["citation_grounded", "avg_grounding_score", "trust_score"],
        var_name="metric",
        value_name="value",
    )
    risk_chart = alt.Chart(risk_melt).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("value:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("model:N", title=None, sort="-x"),
        color=alt.Color("metric:N", title="Metric"),
        column=alt.Column("metric:N", title=None),
        tooltip=["model", "metric", alt.Tooltip("value:Q", format=".3f")],
    ).properties(height=220)
    st.subheader("Grounding and Trust")
    st.caption("Higher is better. `trust_score` is `1 - hallucination_rate` so bigger bars are safer.")
    st.altair_chart(risk_chart, use_container_width=True)

    latency_chart = alt.Chart(summary_df).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("avg_latency_s:Q", title="Average latency (seconds)"),
        y=alt.Y("model:N", title=None, sort="x"),
        color=alt.Color("model:N", legend=None),
        tooltip=["model", alt.Tooltip("avg_latency_s:Q", format=".3f")],
    ).properties(height=220)
    st.subheader("Latency")
    st.caption("Lower is better.")
    st.altair_chart(latency_chart, use_container_width=True)

    heatmap_df = risk_df.melt(
        id_vars=["model"],
        value_vars=[
            "answer_em",
            "answer_contains",
            "citation_grounded",
            "citation_f1",
            "avg_grounding_score",
            "trust_score",
        ],
        var_name="metric",
        value_name="value",
    )
    heatmap = alt.Chart(heatmap_df).mark_rect().encode(
        x=alt.X("metric:N", title=None),
        y=alt.Y("model:N", title=None),
        color=alt.Color("value:Q", title="Score", scale=alt.Scale(domain=[0, 1], scheme="tealblues")),
        tooltip=["model", "metric", alt.Tooltip("value:Q", format=".3f")],
    ).properties(height=220)
    st.subheader("Metric Heatmap")
    st.altair_chart(heatmap, use_container_width=True)


def _render_single_eval(payload: dict) -> None:
    aggregate = payload["aggregate"]
    summary_df = _summary_frame([{"model": payload["model"], **aggregate}])
    _render_eval_overview(summary_df)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("By Difficulty")
        st.dataframe(pd.DataFrame(aggregate.get("by_difficulty", {})).T, use_container_width=True)
    with col_b:
        st.subheader("By Category")
        st.dataframe(pd.DataFrame(aggregate.get("by_category", {})).T, use_container_width=True)

    st.subheader("Per-Case Metrics")
    st.dataframe(pd.DataFrame(payload["cases"]), use_container_width=True)


def _render_multi_eval(payload: dict) -> None:
    summary_df = _summary_frame(payload["summary_rows"])
    _render_eval_overview(summary_df)

    diff_df = _breakdown_frame(payload["aggregates_by_model"], "by_difficulty")
    cat_df = _breakdown_frame(payload["aggregates_by_model"], "by_category")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Difficulty Breakdown")
        st.dataframe(diff_df, use_container_width=True)
    with col_b:
        st.subheader("Category Breakdown")
        st.dataframe(cat_df, use_container_width=True)

    st.subheader("Per-Case Comparison")
    selected_model = st.selectbox("Detailed model view", payload["models"], key="eval_detail_model")
    model_cases = pd.DataFrame(payload["cases_by_model"][selected_model])
    st.dataframe(model_cases, use_container_width=True)

    merged_cases = _merge_case_rows(payload["cases_by_model"])
    st.subheader("Merged Case Table")
    st.dataframe(merged_cases, use_container_width=True)


st.set_page_config(page_title="CodeGraph", layout="wide")
st.title("CodeGraph - grounded Q&A over code")

if "repo_id" not in st.session_state:
    st.session_state.repo_id = None
if "stats" not in st.session_state:
    st.session_state.stats = None
if "ask_result" not in st.session_state:
    st.session_state.ask_result = None
if "eval_result" not in st.session_state:
    st.session_state.eval_result = None
if "graph_result" not in st.session_state:
    st.session_state.graph_result = None

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
    model = st.selectbox("LLM", SUPPORTED_MODELS)

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
            if not _handle_stale_repo(r):
                st.error(r.text)
        else:
            st.session_state.ask_result = r.json()

    if st.session_state.ask_result:
        d = st.session_state.ask_result
        colA, colB = st.columns([2, 1])
        with colA:
            st.subheader("Answer")
            st.write(d["answer"])
            st.caption(f"grounded={d['grounded']} - model={d['model']} - {d['latency_s']:.2f}s")
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
                if "decision" in step:
                    label = list(step["decision"].keys())[0]
                    body = step["decision"]
                elif "guardrail" in step:
                    label = "guardrail"
                    body = step
                elif "re_retrieval" in step:
                    label = "re-retrieval"
                    body = step
                else:
                    label = "info"
                    body = step
                with st.expander(f"step {step.get('step', '?')}: {label}"):
                    st.json(body)

with tab_eval:
    st.write("Run CodeGraphEval across one or more models and compare quality, grounding, and latency.")
    eval_path = st.text_input("Eval JSON", value="data/CodeGraphEval_50.json")
    eval_models = st.multiselect(
        "Models to compare",
        SUPPORTED_MODELS,
        default=[model],
    )
    max_cases = st.number_input("Max cases (0 = all)", min_value=0, value=0)
    run_ablation = st.checkbox("Run Ablation Study (Graph RAG vs. Baseline RAG)")
    button_label = "Run ablation study" if run_ablation else ("Run comparison" if len(eval_models) > 1 else "Run evaluation")
    if st.button(button_label):
        if not eval_models and not run_ablation:
            st.error("Select at least one model.")
        else:
            body = {"eval_path": eval_path, "run_ablation_study": run_ablation}
            if len(eval_models) == 1:
                body["model"] = eval_models[0]
            elif len(eval_models) > 1:
                body["models"] = eval_models
            else:
                body["model"] = model # default if nothing selected but running ablation
            
            if max_cases:
                body["max_cases"] = int(max_cases)
            with st.spinner("Evaluating..."):
                r = requests.post(f"{API}/evaluate", json=body, timeout=3600)
            if not r.ok:
                st.error(r.text)
            else:
                st.session_state.eval_result = r.json()
    
    if st.session_state.eval_result:
        payload = st.session_state.eval_result
        if "summary_rows" in payload:
            _render_multi_eval(payload)
        else:
            _render_single_eval(payload)

with tab_graph:
    name = st.text_input("Symbol (function/class) name to view neighborhood")
    radius = st.slider("Radius", 1, 3, 1)
    if st.button("Show graph") and name and st.session_state.repo_id:
        r = requests.get(f"{API}/graph/{st.session_state.repo_id}/{name}", params={"radius": radius})
        if r.ok:
            st.session_state.graph_result = r.json()
        else:
            if not _handle_stale_repo(r):
                st.error(r.text)

    if st.session_state.graph_result:
        data = st.session_state.graph_result
        try:
            from streamlit_agraph import agraph, Node, Edge, Config
            nodes = [Node(id=n["id"], label=n.get("name", n["id"])) for n in data["nodes"]]
            edges = [Edge(source=e["source"], target=e["target"], label=e.get("type", "")) for e in data["edges"]]
            agraph(nodes=nodes, edges=edges, config=Config(width=900, height=600, directed=True))
        except Exception:
            st.json(data)
