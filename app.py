import os
import streamlit as st
import tempfile
from rag_pipeline import (
    load_and_split,
    build_vectorstore,
    load_vectorstore,
    build_qa_chain,
    ask_question,
)

st.set_page_config(page_title="DocChat — RAG Q&A", page_icon="📄", layout="wide")
st.title(" DocChat — Ask Questions About Your PDF")
st.caption("Powered by LangChain · ChromaDB · Gemini 2.5 Flash · HuggingFace Embeddings")

if "qa_chain"     not in st.session_state: st.session_state.qa_chain     = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "doc_name"     not in st.session_state: st.session_state.doc_name     = None

with st.sidebar:
    st.header("1. Upload your PDF")
    uploaded = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded:
        tmp_path = os.path.join(tempfile.gettempdir(), uploaded.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())

        if st.button(" Index document", use_container_width=True):
            with st.spinner("Splitting and embedding… (takes ~30s first time)"):
                st.session_state.qa_chain = None
                chunks = load_and_split(tmp_path)
                vectordb, collection = build_vectorstore(chunks)
                st.session_state.qa_chain = build_qa_chain(vectordb)
                st.session_state.doc_name = uploaded.name
                st.session_state.chat_history = []
            st.success(f" Indexed {len(chunks)} chunks from **{uploaded.name}**")

    st.divider()
    st.header("2. Or load existing index")
    if os.path.exists("./chroma_db"):
        if st.button(" Load saved index", use_container_width=True):
            with st.spinner("Loading from disk..."):
                vectordb = load_vectorstore()
                st.session_state.qa_chain = build_qa_chain(vectordb)
            st.success("Loaded!")
    else:
        st.caption("No saved index yet. Upload a PDF first.")

    st.divider()
    show_sources  = st.toggle("Show source chunks", value=True)
    show_snippets = st.toggle("Show text snippets", value=False)

if st.session_state.qa_chain is None:
    st.info(" Open the sidebar, upload a PDF and click **Index document** to start.")
    st.stop()

st.subheader(f"Chatting with: {st.session_state.doc_name or 'your document'}")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            with st.expander(f" {len(msg['sources'])} source(s)"):
                for src in msg["sources"]:
                    st.markdown(f"**Page {src['page']}**")
                    if show_snippets:
                        st.caption(f"> {src['snippet']}…")

question = st.chat_input("Ask anything about your PDF…")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result  = ask_question(st.session_state.qa_chain, question)
        answer  = result["answer"]
        sources = result["sources"]
        st.markdown(answer)
        if show_sources and sources:
            with st.expander(f" {len(sources)} source(s)"):
                for src in sources:
                    st.markdown(f"**Page {src['page']}**")
                    if show_snippets:
                        st.caption(f"> {src['snippet']}…")

    st.session_state.chat_history.append({
        "role": "assistant", "content": answer, "sources": sources,
    })

if st.session_state.chat_history:
    if st.button(" Clear chat"):
        st.session_state.chat_history = []
        st.rerun()