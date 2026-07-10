import os
import tempfile
import uuid
import streamlit as st

from rag_pipeline import load_and_split, build_vectorstore, load_vectorstore
from retrieval import build_hybrid_retriever
from tools import make_document_search_tool, web_search, calculator
from agent import build_agent, run_agent

st.set_page_config(page_title="Agentic RAG", page_icon="", layout="wide")
st.title(" Agentic RAG — Document + Web + Math Assistant")
st.caption("Powered by LangGraph · Hybrid Retrieval (BM25 + Vector) · Cross-Encoder Reranking · Gemini 2.5 Flash")

# --- Session state ---
if "agent_app" not in st.session_state: st.session_state.agent_app = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "doc_name" not in st.session_state: st.session_state.doc_name = None
if "thread_id" not in st.session_state: st.session_state.thread_id = str(uuid.uuid4())


def _init_agent(vectordb, chunks):
    retriever = build_hybrid_retriever(vectordb, chunks)
    doc_tool = make_document_search_tool(retriever)
    return build_agent([doc_tool, web_search, calculator])


with st.sidebar:
    st.header("1. Upload your PDF")
    uploaded = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded and st.button(" Index document", use_container_width=True):
        with st.spinner("Splitting and embedding… (~30s first time)"):
            tmp_path = os.path.join(tempfile.gettempdir(), uploaded.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())

            chunks = load_and_split(tmp_path)
            vectordb, _collection = build_vectorstore(chunks)
            st.session_state.agent_app = _init_agent(vectordb, chunks)
            st.session_state.doc_name = uploaded.name
            st.session_state.chat_history = []
            st.session_state.thread_id = str(uuid.uuid4())  # fresh memory for a new doc
        st.success(f"Indexed {len(chunks)} chunks from **{uploaded.name}**")

    st.divider()
    st.header("2. Settings")
    show_trace = st.toggle("Show agent reasoning trace", value=True)

    if st.button(" Reset conversation memory", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.success("Memory cleared — starting a new conversation thread.")

if st.session_state.agent_app is None:
    st.info(" Upload a PDF and click **Index document** to start. You can also just ask general or math questions once indexed — the agent will route to the web or calculator as needed.")
    st.stop()

st.subheader(f"Chatting about: {st.session_state.doc_name}")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_trace and msg.get("trace"):
            with st.expander(f" Reasoning trace ({len(msg['trace'])} step(s))"):
                for step in msg["trace"]:
                    st.markdown(step)

question = st.chat_input("Ask about your document, the web, or do some math…")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            messages = run_agent(st.session_state.agent_app, question, thread_id=st.session_state.thread_id)

        # Build a readable trace from tool calls + tool results in this turn
        trace = []
        for m in messages:
            if getattr(m, "tool_calls", None):
                for call in m.tool_calls:
                    trace.append(f" Called `{call['name']}` with `{call['args']}`")
            elif hasattr(m, "tool_call_id"):
                preview = m.content[:200] + ("…" if len(m.content) > 200 else "")
                trace.append(f" `{m.name}` returned: {preview}")

        final_answer = messages[-1].content
        st.markdown(final_answer)
        if show_trace and trace:
            with st.expander(f" Reasoning trace ({len(trace)} step(s))"):
                for step in trace:
                    st.markdown(step)

    st.session_state.chat_history.append({"role": "assistant", "content": final_answer, "trace": trace})

if st.session_state.chat_history:
    if st.button(" Clear chat display"):
        st.session_state.chat_history = []
        st.rerun()