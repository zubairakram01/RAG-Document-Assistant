import os
import streamlit as st
from src.document_loader import load_pdf
from src.text_splitter import split_documents
from src.vector_store import create_vector_store
from src.rag_pipeline import run_rag_pipeline

# Page configuration
st.set_page_config(
    page_title="RAG Document Intelligence Assistant",
    page_icon="📚",
    layout="wide",
)

# Ensure documents directory exists
UPLOAD_DIR = os.path.join("data", "documents")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# Header
st.title("📚 RAG Document Intelligence Assistant")
st.caption(
    "Built with LangChain (LCEL), ChromaDB, HuggingFace Embeddings & Google Gemini — "
    "featuring Basic RAG, Multi-Query Retrieval, and RAG-Fusion."
)

# Sidebar Controls
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    with st.expander("⚙️ Chunking & Retrieval Settings", expanded=False):
        chunk_size = st.slider("Chunk Size", min_value=300, max_value=2000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
        top_k = st.slider("Top-K Retrieved Chunks", min_value=2, max_value=8, value=4, step=1)

    if st.button("🚀 Process Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file first.")
        else:
            with st.spinner("Loading PDFs, splitting chunks, and building Chroma vector index..."):
                try:
                    all_docs = []
                    saved_names = []
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        docs = load_pdf(file_path)
                        all_docs.extend(docs)
                        saved_names.append(uploaded_file.name)

                    chunks = split_documents(
                        all_docs,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                    st.session_state.vector_store = create_vector_store(chunks)
                    st.session_state.processed_files = saved_names
                    st.session_state.chat_history = []
                    st.success(
                        f"Indexed {len(saved_names)} file(s) into {len(chunks)} chunks!"
                    )
                except Exception as e:
                    st.error(f"Error processing documents: {e}")

    st.divider()
    st.header("2. RAG Strategy")
    rag_mode = st.radio(
        "Select Retrieval Technique:",
        options=["Basic RAG", "Multi-Query RAG", "RAG-Fusion"],
        index=2,
        help=(
            "• Basic RAG: Standard vector similarity search.\n"
            "• Multi-Query RAG: Generates 3 alternative query perspectives.\n"
            "• RAG-Fusion: Multi-Query + Reciprocal Rank Fusion (RRF) re-ranking."
        ),
    )

    if st.session_state.processed_files:
        st.divider()
        st.markdown("**Active Documents:**")
        for fname in st.session_state.processed_files:
            st.markdown(f"- `{fname}`")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main Chat Interface
if st.session_state.vector_store is None:
    st.info("👈 Upload your PDF document(s) in the sidebar and click **Process Documents** to begin.")
else:
    # Display previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if len(msg.get("queries", [])) > 1:
                    with st.expander(f"🔍 Generated Search Queries ({msg['mode']})"):
                        for idx, q in enumerate(msg["queries"], start=1):
                            st.markdown(f"{idx}. {q}")

                if msg.get("sources"):
                    with st.expander("📄 Retrieved Source Chunks & Citations"):
                        for idx, doc in enumerate(msg["sources"], start=1):
                            src = doc.metadata.get("source", "Unknown")
                            page = doc.metadata.get("page", 0) + 1
                            st.markdown(f"**Chunk {idx}** — `{src}` (Page {page})")
                            st.caption(doc.page_content)
                            st.divider()

    # User input box
    user_question = st.chat_input("Ask a question about your uploaded PDF(s)...")

    if user_question:
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Run RAG pipeline and display assistant response
        with st.chat_message("assistant"):
            with st.spinner(f"Retrieving context using {rag_mode} & generating answer..."):
                try:
                    result = run_rag_pipeline(
                        vector_store=st.session_state.vector_store,
                        question=user_question,
                        mode=rag_mode,
                        top_k=top_k,
                    )
                    st.markdown(result["answer"])

                    if len(result["queries"]) > 1:
                        with st.expander(f"🔍 Generated Search Queries ({result['mode']})"):
                            for idx, q in enumerate(result["queries"], start=1):
                                st.markdown(f"{idx}. {q}")

                    if result["sources"]:
                        with st.expander("📄 Retrieved Source Chunks & Citations"):
                            for idx, doc in enumerate(result["sources"], start=1):
                                src = doc.metadata.get("source", "Unknown")
                                page = doc.metadata.get("page", 0) + 1
                                st.markdown(f"**Chunk {idx}** — `{src}` (Page {page})")
                                st.caption(doc.page_content)
                                st.divider()

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "queries": result["queries"],
                            "sources": result["sources"],
                            "mode": result["mode"],
                        }
                    )
                except Exception as e:
                    st.error(f"Pipeline Error: {e}")