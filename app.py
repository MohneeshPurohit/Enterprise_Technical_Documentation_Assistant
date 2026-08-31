"""
Phase 12: Streamlit Interactive Web Application UI
===================================================
Enterprise UI for the Technical Documentation Assistant RAG System.
"""

import sys, os
import time
import json
import streamlit as st

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.document_loader import DocumentLoader
from src.chunking import TechnicalChunker
from src.retriever import TechnicalRetriever
from src.rag_pipeline import TechnicalRAGPipeline

st.set_page_config(
    page_title="Technical Documentation Assistant | AI Capstone",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper Function: Initialize Pipeline
@st.cache_resource
def get_pipeline():
    try:
        retriever = TechnicalRetriever.load("vectorstore")
    except Exception:
        loader = DocumentLoader("data/documents")
        docs = loader.load_all_documents()
        chunker = TechnicalChunker(chunk_size=600, chunk_overlap=100)
        chunks = chunker.chunk_documents(docs)
        retriever = TechnicalRetriever()
        retriever.build_index(chunks)
        retriever.save("vectorstore")
    
    return TechnicalRAGPipeline("vectorstore", retriever=retriever)

pipeline = get_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Controls
with st.sidebar:
    st.title("System Controls")
    st.markdown("---")

    # Document Uploader
    st.subheader("1. Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload Technical Docs (PDF, TXT, MD)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            save_path = os.path.join("data/documents", file.name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s) to `data/documents/`.")
        if st.button("Index Uploaded Documents"):
            with st.spinner("Re-building FAISS Vector Index..."):
                loader = DocumentLoader("data/documents")
                docs = loader.load_all_documents()
                chunker = TechnicalChunker(chunk_size=600, chunk_overlap=100)
                chunks = chunker.chunk_documents(docs)
                pipeline.retriever.build_index(chunks)
                pipeline.retriever.save("vectorstore")
                st.cache_resource.clear()
            st.success("Vector Index successfully updated!")
            st.rerun()

    st.markdown("---")
    st.subheader("2. Vector DB Status")
    num_chunks = pipeline.retriever.index.ntotal if pipeline.retriever.index else 0
    st.info(f"**FAISS Indexed Chunks:** `{num_chunks}`\n\n**Embedding Model:** `all-MiniLM-L6-v2`\n\n**LLM Grounding:** Strict Context Only")

    st.markdown("---")
    if st.button("Clear Chat History", type="primary"):
        st.session_state.messages = []
        st.rerun()

# Main App Tabs
st.title("Enterprise Technical Documentation Assistant")

tab_chat, tab_eval, tab_docs = st.tabs(["💬 AI Support Chat", "📊 Capstone Benchmark Evaluation", "📁 Document Library"])

with tab_chat:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "retrieved_chunks" in message and message["retrieved_chunks"]:
                with st.expander("🔍 View Retrieved Documentation Context & Sources"):
                    for idx, chunk in enumerate(message["retrieved_chunks"], start=1):
                        meta = chunk["metadata"]
                        st.markdown(f"**Snippet #{idx}** — *{meta['source']}* (Page {meta['page']}) | Similarity Score: `{chunk.get('similarity_score', 'N/A')}`")
                        st.code(chunk["content"], language="text")

    prompt_input = st.chat_input("Ask a question about API endpoints, errors, authentication, or webhooks...")
    if prompt_input:
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching FAISS vector DB & synthesizing grounded response..."):
                start_time = time.time()
                res = pipeline.run(prompt_input, chat_history=st.session_state.messages[:-1])
                latency = round(time.time() - start_time, 2)

                st.markdown(res["answer"])

                if res["retrieved_chunks"]:
                    with st.expander(f"🔍 View Retrieved Context & Sources ({latency}s latency)"):
                        for idx, chunk in enumerate(res["retrieved_chunks"], start=1):
                            meta = chunk["metadata"]
                            st.markdown(f"**Snippet #{idx}** — *{meta['source']}* (Page {meta['page']}) | Similarity Score: `{chunk.get('similarity_score', 'N/A')}`")
                            st.code(chunk["content"], language="text")

        st.session_state.messages.append({
            "role": "assistant",
            "content": res["answer"],
            "retrieved_chunks": res["retrieved_chunks"]
        })

with tab_eval:
    st.subheader("Quantitative RAG Evaluation Benchmark Results")
    if os.path.exists("evaluation_results.json"):
        with open("evaluation_results.json", "r") as f:
            eval_data = json.load(f)

        col1, col2, col3 = st.columns(3)
        col1.metric("Retrieval Hit Rate @ K", f"{eval_data['retrieval_hit_rate']}%")
        col2.metric("Answer Groundedness Accuracy", f"{eval_data['answer_accuracy_rate']}%")
        col3.metric("Hallucination Refusal Rate", f"{eval_data['hallucination_refusal_rate']}%")

        st.dataframe(eval_data["test_cases"], use_container_width=True)

with tab_docs:
    st.subheader("Ingested Technical Documentation Collection")
    loader = DocumentLoader("data/documents")
    raw_docs = loader.load_all_documents()
    for doc in raw_docs:
        meta = doc["metadata"]
        with st.expander(f"📄 {meta['source']} (Page {meta['page']} of {meta['total_pages']})"):
            st.code(doc["content"], language="text")