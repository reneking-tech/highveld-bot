from typing import Optional
import os

from app.config import INDEX_DIR, EMBEDDING_MODEL, TOP_K  # paths & knobs
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Cached vector store (loaded once on first call)
_vectordb: Optional[FAISS] = None


def load_vectorstore() -> FAISS:
    """Load the persisted FAISS index from disk and cache it.
    Assumes ingest has saved a FAISS index to INDEX_DIR.
    """
    global _vectordb
    if _vectordb is not None:
        return _vectordb

    if not os.path.isdir(INDEX_DIR):
        raise FileNotFoundError(
            f"Vector index not found at {INDEX_DIR}. Create it during ingest (FAISS.save_local)."
        )

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    _vectordb = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return _vectordb


def get_retriever(top_k: int = TOP_K):
    """Return a retriever over the vector store using MMR.
    MMR improves diversity and reduces redundancy among results.
    `top_k` controls how many chunks are returned to the RAG chain.
    """
    vectordb = load_vectorstore()
    fetch_k = max(8, int(top_k) * 4)
    return vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": int(top_k),
            "fetch_k": fetch_k,
            "lambda_mult": 0.5,
        },
    )


def top_similarity_distance(query: str) -> float | None:
    """Best single-match distance for a query (lower is better for cosine).
    Returns None if scores are unavailable.
    """
    try:
        vectordb = load_vectorstore()
        res = vectordb.similarity_search_with_score(query, k=1)
        if not res:
            return None
        _doc, score = res[0]
        try:
            return float(score)
        except Exception:
            return None
    except Exception:
        return None
