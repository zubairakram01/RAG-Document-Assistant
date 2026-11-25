import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str) -> List[Document]:
    """Load a single PDF file and return a list of Document objects with metadata."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Clean up source metadata to show just the filename
    filename = os.path.basename(file_path)
    for doc in documents:
        doc.metadata["source"] = filename

    return documents