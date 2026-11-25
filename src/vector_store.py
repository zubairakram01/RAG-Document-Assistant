import uuid
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.embeddings import get_embedding_model


def create_vector_store(chunks: List[Document]) -> Chroma:
    """Create a fresh Chroma vector store from document chunks."""
    embeddings = get_embedding_model()
    collection_name = f"rag_docs_{uuid.uuid4().hex[:8]}"

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
    )
    return vector_store