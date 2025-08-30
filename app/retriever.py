from typing import Optional
import os

from app.config import INDEX_DIR, EMBEDDING_MODEL, TOP_K  # paths & knobs

# Optional dependencies: provide safe fallbacks if unavailable
try:  # LangChain embeddings
    from langchain_openai import OpenAIEmbeddings  # type: ignore
except Exception:  # pragma: no cover
    OpenAIEmbeddings = None  # type: ignore

try:  # FAISS vector store
    from langchain_community.vectorstores import FAISS  # type: ignore
except Exception:  # pragma: no cover
    FAISS = None  # type: ignore

# Cached vector store (loaded once on first call)
_vectordb: Optional[object] = None


def _emb():
    if OpenAIEmbeddings is None:
        raise RuntimeError("OpenAIEmbeddings not available (langchain_openai missing)")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    # Prefer OPENAI_BASE_URL, then legacy OPENAI_API_BASE, else default
    base = (
        os.getenv("OPENAI_BASE_URL", "")
        or os.getenv("OPENAI_API_BASE", "")
        or "https://api.openai.com/v1"
    ).strip()
    # Drop blank env that can confuse SDKs
    if not base:
        base = "https://api.openai.com/v1"
    if not base.startswith(("http://", "https://")):
        base = f"https://{base}"
    # Normalise common host without path
    # Both openai>=1.x and langchain_openai accept base_url with /v1
    kw = {"model": EMBEDDING_MODEL, "base_url": base}  # type: ignore
    if api_key:
        kw["api_key"] = api_key  # type: ignore
    return OpenAIEmbeddings(**kw)  # type: ignore[arg-type]


def load_vectorstore():
    """Load the persisted FAISS index from disk and cache it.
    Assumes ingest has saved a FAISS index to INDEX_DIR.
    """
    global _vectordb
    if _vectordb is not None:
        return _vectordb

    # If FAISS or embeddings are unavailable, return a no-op store
    if FAISS is None or OpenAIEmbeddings is None:
        _vectordb = _NoopVectorStore()
        return _vectordb

    if not os.path.isdir(INDEX_DIR):
        # Fall back to a no-op store rather than crashing import paths
        _vectordb = _NoopVectorStore()
        return _vectordb

    embeddings = _emb()
    _vectordb = FAISS.load_local(  # type: ignore[call-arg,attr-defined]
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
    if isinstance(vectordb, _NoopVectorStore):
        return _NoopRetriever()
    fetch_k = max(8, int(top_k) * 4)
    return vectordb.as_retriever(  # type: ignore[return-value]
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
        if isinstance(vectordb, _NoopVectorStore):
            return None
        res = vectordb.similarity_search_with_score(query, k=1)  # type: ignore[attr-defined]
        if not res:
            return None
        _doc, score = res[0]
        return float(score)
    except Exception:
        return None


# ---------------- No-op fallbacks (safe without external deps) ---------------- #


class _NoopRetriever:
    def invoke(self, query: str):  # matches retriever.invoke API
        return []


class _NoopVectorStore:
    def as_retriever(self, *_, **__):
        return _NoopRetriever()

    def similarity_search_with_score(self, *_args, **_kwargs):
        return []
