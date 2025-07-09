from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever
import os

# Function to load and index documents from a specified directory
# with nomic-embedding, MMR search, and fetch_k/k validation

def load_and_index_documents_by_file(
    path=r'C:\Users\dell\Downloads\new-Task-Coreali\MPB',
    k: int = 3,
    fetch_k: int = 6
) -> VectorStoreRetriever:
    docs = []
    for fname in os.listdir(path):
        file_path = os.path.join(path, fname)
        with open(file_path, "rb") as f:
            raw = f.read()
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                content = raw.decode("latin-1")

        docs.append(Document(page_content=content, metadata={"source": fname}))

    # Use nomic-embedding model
    embedding_model = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    model_kwargs={"trust_remote_code": True})

    # Build FAISS index
    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local("faiss_index_by_file")

    # Ensure fetch_k >= k
    if fetch_k < k:
        raise ValueError(f"fetch_k ({fetch_k}) must be >= k ({k})")

    # Create the retriever with MMR search
    retriever: VectorStoreRetriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k
        }
    )

    return retriever
