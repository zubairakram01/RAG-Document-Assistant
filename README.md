# 📚 RAG Document Intelligence Assistant

An end-to-end **Retrieval-Augmented Generation (RAG)** web application built with **LangChain (LCEL)**, **ChromaDB**, **HuggingFace Embeddings**, **Groq LLM**, and **Streamlit**.

This application enables users to upload PDF documents, index them into a vector database, and ask complex natural-language questions using advanced retrieval strategies (**Basic RAG**, **Multi-Query Retrieval**, and **RAG-Fusion** with Reciprocal Rank Fusion).

---

## ✨ Key Features

- **Multi-PDF Ingestion & Chunking:** Loads PDF documents via `PyPDFLoader` and splits text using `RecursiveCharacterTextSplitter` with configurable chunk size and overlap.
- **Local Semantic Embeddings:** Uses HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` locally for fast, zero-cost vectorization.
- **Vector Storage:** Stores and queries embeddings dynamically using **ChromaDB**.
- **3 Selectable Retrieval Strategies:**
  1. **Basic RAG:** Standard dense vector similarity search.
  2. **Multi-Query RAG:** Uses an LLM to generate multiple alternative perspectives of the user's question to overcome distance-based search limitations.
  3. **RAG-Fusion:** Combines Multi-Query generation with **Reciprocal Rank Fusion (RRF)** mathematical re-ranking to prioritize the most consistently relevant document chunks.
- **Grounded Answers & Source Citations:** Enforces strict context adherence to prevent hallucinations and displays exact source filenames, page numbers, and generated sub-queries in the UI.

---

## 🏗️ Project Architecture

```text
RAG-Document-Assistant/
├── data/
│   └── documents/          # Uploaded PDF storage (ignored in git)
├── src/
│   ├── __init__.py
│   ├── document_loader.py  # PDF ingestion and metadata extraction
│   ├── text_splitter.py    # Recursive character text chunking
│   ├── embeddings.py       # Local HuggingFace embedding model loader
│   ├── vector_store.py     # ChromaDB vector store creation
│   ├── retriever.py        # Basic, Multi-Query, and RAG-Fusion (RRF) logic
│   └── rag_pipeline.py     # LCEL chain & grounded LLM response generation
├── app.py                  # Streamlit interactive user interface
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation