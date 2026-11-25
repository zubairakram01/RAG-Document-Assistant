import re
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma


def generate_multi_queries(question: str, llm) -> List[str]:
    """Generate 3 alternative perspectives of the user's question (from RAG From Scratch)."""
    prompt = ChatPromptTemplate.from_template(
        """You are an AI language model assistant. Your task is to generate 3 different 
versions of the given user question to retrieve relevant documents from a vector database. 
By generating multiple perspectives on the user question, your goal is to help overcome 
some of the limitations of distance-based similarity search. 
Provide these alternative questions separated by newlines without numbering or bullet points.

Original question: {question}"""
    )

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question})

    queries = []
    for line in response.strip().split("\n"):
        cleaned = re.sub(r"^[\d\.\-\*\)\s]+", "", line).strip()
        if cleaned:
            queries.append(cleaned)

    # Always include the original question first
    all_queries = [question] + queries[:3]
    return list(dict.fromkeys(all_queries))


def reciprocal_rank_fusion(
    results_list: List[List[Document]], k: int = 60, top_n: int = 4
) -> List[Document]:
    """Re-rank retrieved documents across multiple queries using Reciprocal Rank Fusion (RRF)."""
    fused_scores = {}
    doc_map = {}

    for docs in results_list:
        for rank, doc in enumerate(docs):
            doc_key = (doc.page_content, doc.metadata.get("page", 0))
            if doc_key not in doc_map:
                doc_map[doc_key] = doc
                fused_scores[doc_key] = 0.0
            fused_scores[doc_key] += 1.0 / (rank + k)

    reranked_keys = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    return [doc_map[key] for key in reranked_keys[:top_n]]


def retrieve_documents(
    vector_store: Chroma,
    question: str,
    llm,
    mode: str = "Basic RAG",
    top_k: int = 4,
) -> Tuple[List[Document], List[str]]:
    """
    Retrieve relevant documents based on selected RAG strategy:
    - 'Basic RAG': Standard vector similarity search
    - 'Multi-Query RAG': Generates multiple query variations & deduplicates results
    - 'RAG-Fusion': Generates multiple query variations & re-ranks via RRF
    """
    base_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    if mode == "Basic RAG":
        docs = base_retriever.invoke(question)
        return docs, [question]

    # Generate multiple query variations for Multi-Query & RAG-Fusion
    queries = generate_multi_queries(question, llm)
    all_results = [base_retriever.invoke(q) for q in queries]

    if mode == "Multi-Query RAG":
        unique_docs = []
        seen = set()
        for docs in all_results:
            for doc in docs:
                doc_key = (doc.page_content, doc.metadata.get("page", 0))
                if doc_key not in seen:
                    seen.add(doc_key)
                    unique_docs.append(doc)
        return unique_docs[:top_k], queries

    # RAG-Fusion (Reciprocal Rank Fusion)
    fused_docs = reciprocal_rank_fusion(all_results, top_n=top_k)
    return fused_docs, queries