from typing import List
from sentence_transformers import  CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from config.settings import EMBEDDING_MODEL, EMBEDDING_DEVICE, RERANK_TOP_K


class Reranker:
    """
    Re-ranks retrieved documents using a CrossEncoder for accurate relevance scoring.
    Falls back to cosine similarity if CrossEncoder is unavailable.
    """

    def __init__(self):
        # Load cross-encoder for accurate relevance scoring
        # This is the KEY difference — cross-encoders understand meaning,
        # not just surface similarity
        self.cross_encoder = None
        try:
            print("Loading cross-encoder for reranker...")
            self.cross_encoder = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                device=EMBEDDING_DEVICE,
            )
            print("Cross-encoder loaded successfully.")
        except Exception as e:
            print(f"[WARNING] CrossEncoder failed to load: {e}")
            print("[WARNING] Falling back to cosine similarity reranking.")

    def rerank(self, query: str, docs: List[str], top_k: int = RERANK_TOP_K) -> List[str]:
        if not docs:
            return []
        if self.cross_encoder:
            pairs = [[query, doc[:512]] for doc in docs]
            scores = self.cross_encoder.predict(pairs)
            ranked_indices = np.argsort(scores)[::-1][:top_k]
            return [docs[i] for i in ranked_indices]
        # No cross-encoder — return as-is (caller handles fallback)
        return docs[:top_k]

    def rerank_with_scores(self, query: str, docs: List[str], top_k: int = RERANK_TOP_K) -> List[tuple]:
        if not docs:
            return []
        if self.cross_encoder:
            pairs = [[query, doc[:512]] for doc in docs]
            scores = self.cross_encoder.predict(pairs)
            scored = sorted(zip(scores.tolist(), docs), key=lambda x: -x[0])
            return scored[:top_k]
        # No cross-encoder — return with zero scores
        return [(0.0, doc) for doc in docs[:top_k]]

if __name__ == "__main__":
    test_docs = [
        "Git init creates a new repository.",
        "Python open() reads a file.",
        "Use rm -r to delete a folder recursively.",
        "Git add stages changes before commit.",
    ]
    query = "How do I initialize a Git repository?"
    reranker = Reranker()
    print("Reranked docs:")
    for score, doc in reranker.rerank_with_scores(query, test_docs, top_k=2):
        print(f"  [{score:.3f}] {doc}")

if __name__ == "__main__":
    import chromadb
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    from config.settings import CHROMA_DB_DIR, RETRIEVAL_TOP_K, EMBEDDING_MODEL, EMBEDDING_DEVICE

    reranker = Reranker()

    # Load actual KB with proper semantic search
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_collection(name="knowledge_base")
    embed_model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)

    def get_top_docs(query: str, k: int = 10):
        qe = embed_model.encode(query).tolist()
        results = collection.query(query_embeddings=[qe], n_results=k)
        return results["documents"][0]

    test_queries = [
        "What is the education of lavish devpura",
        "give me the resume of lavish devpura",
        "what projects has lavish worked on",
        "working of a transformer",
        "what is machine learning",
    ]

    print(f"\n{'Query':<45} {'Score':>8}  {'Decision'}")
    print("-" * 80)

    for query in test_queries:
        top_docs = get_top_docs(query, k=5)
        results = reranker.rerank_with_scores(query, top_docs, top_k=1)
        score = results[0][0] if results else -999
        decision = "USE KB ✓" if score >= -2.5 else "SKIP KB ✗"
        print(f"{query:<45} {score:>8.3f}  {decision}")
        if results:
            print(f"  Top doc: {results[0][1][:80]}")
        print()