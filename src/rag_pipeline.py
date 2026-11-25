import os
import json
import urllib.request
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from src.retriever import retrieve_documents

load_dotenv()

_CACHED_GROQ_MODEL = None


def _get_active_groq_model(api_key: str) -> str:
    """Automatically fetch the best active chat model from Groq's live models endpoint."""
    global _CACHED_GROQ_MODEL
    if _CACHED_GROQ_MODEL:
        return _CACHED_GROQ_MODEL

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0",
        },
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    # Filter out audio, TTS, and moderation/guard models
    excluded = ("whisper", "guard", "tts", "orpheus", "playai", "distil-whisper")
    chat_models = [
        m["id"]
        for m in data.get("data", [])
        if not any(bad in m["id"].lower() for bad in excluded)
    ]

    if not chat_models:
        raise ValueError("No active chat models found on your Groq account.")

    # Prefer strong general-purpose LLMs in priority order
    preferred_keywords = ("llama-4", "llama-3.3", "qwen", "gpt-oss", "llama-3.1", "gemma")
    for keyword in preferred_keywords:
        for model_id in chat_models:
            if keyword in model_id.lower():
                _CACHED_GROQ_MODEL = model_id
                return _CACHED_GROQ_MODEL

    _CACHED_GROQ_MODEL = chat_models[0]
    return _CACHED_GROQ_MODEL


def get_llm() -> ChatGroq:
    """Initialize Groq LLM using an active free-tier model."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Please add it to your .env file.")

    model_name = _get_active_groq_model(api_key)
    return ChatGroq(
        model=model_name,
        temperature=0.2,
        api_key=api_key,
    )


def format_docs_with_citations(docs: List[Document]) -> str:
    """Format retrieved chunks with filename and page numbers for grounded generation."""
    formatted_chunks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0) + 1  # Convert 0-indexed page to 1-indexed
        formatted_chunks.append(
            f"[Chunk {i} | Source: {source}, Page: {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted_chunks)


def run_rag_pipeline(
    vector_store: Chroma,
    question: str,
    mode: str = "Basic RAG",
    top_k: int = 4,
) -> Dict[str, Any]:
    """Execute end-to-end RAG retrieval and answer generation."""
    llm = get_llm()

    # Step 1: Retrieve relevant chunks using chosen strategy
    retrieved_docs, generated_queries = retrieve_documents(
        vector_store=vector_store,
        question=question,
        llm=llm,
        mode=mode,
        top_k=top_k,
    )

    context = format_docs_with_citations(retrieved_docs)

    # Step 2: Grounded Generation Prompt (LCEL)
    prompt = ChatPromptTemplate.from_template(
        """You are an intelligent Document Assistant. Answer the user's question strictly 
using the provided context below. If the answer is not present in the context, clearly 
state that the uploaded document does not contain that information—do not make up facts.
When possible, mention the page number(s) where the information was found.

Context:
{context}

Question:
{question}

Answer:"""
    )

    rag_chain = prompt | llm | StrOutputParser()
    answer = rag_chain.invoke({"context": context, "question": question})

    return {
        "answer": answer,
        "sources": retrieved_docs,
        "queries": generated_queries,
        "mode": mode,
    }