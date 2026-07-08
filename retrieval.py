"""
Hybrid retrieval: combines keyword search (BM25) with semantic search
(your existing Chroma vectorstore), then re-ranks the combined results
with a cross-encoder for better precision than either method alone.

Why hybrid: vector search is great at "meaning" but can miss exact terms
(names, codes, numbers). BM25 is great at exact terms but misses
paraphrases. Combining both and re-ranking gets the best of each.
"""
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

# Loaded once and reused — this model is small (~80MB) and runs on CPU fine.
_cross_encoder = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def build_hybrid_retriever(vectordb, chunks: list, k: int = 8):
    """
    vectordb: your existing Chroma instance (from rag_pipeline.build_vectorstore)
    chunks:   the same Document chunks used to build vectordb, needed for BM25
    k:        how many candidates each retriever contributes before re-ranking
    """
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    vector_retriever = vectordb.as_retriever(search_kwargs={"k": k})

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6],  # trust semantic search slightly more
    )
    return ensemble


def rerank(query: str, docs: list, top_n: int = 4) -> list:
    """
    Cross-encoder re-ranking: scores each (query, doc) pair jointly,
    which is more accurate than comparing separately-computed embeddings,
    at the cost of being slower — fine since we only rerank a small
    candidate set (the output of hybrid retrieval), not the whole corpus.
    """
    if not docs:
        return []

    model = _get_cross_encoder()
    pairs = [[query, doc.page_content] for doc in docs]
    scores = model.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _score in ranked[:top_n]]


def hybrid_search(retriever, query: str, top_n: int = 4) -> list:
    """One call: retrieve candidates from both methods, then rerank."""
    candidates = retriever.invoke(query)
    return rerank(query, candidates, top_n=top_n)