import os
import pickle

import faiss
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import (
    ARTIFACTS_DIR,
    DOCUMENTS_PATH,
    BM25_INDEX_DIR,
    DOC_IDS_PATH,
    FAISS_INDEX_PATH,
    BI_ENCODER_MODEL,
    CROSS_ENCODER_MODEL,
    BM25_CANDIDATE_K,
    RRF_K,
    RERANK_K,
    FINAL_K,
)


class RetrievalPipeline:
    def __init__(self, load_reranker=True):
        required_paths = [
            DOCUMENTS_PATH,
            BM25_INDEX_DIR,
            DOC_IDS_PATH,
            FAISS_INDEX_PATH,
        ]

        for path in required_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing {path}. Run export_artifacts.py and build_indexes.py first."
                )

        with open(DOCUMENTS_PATH, "rb") as f:
            self.documents = pickle.load(f)

        with open(DOC_IDS_PATH, "rb") as f:
            self.doc_ids = pickle.load(f)

        self.searcher = LuceneSearcher(BM25_INDEX_DIR)
        self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        self.bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL) if load_reranker else None

    def bm25_search(self, query, k=FINAL_K):
        if not query or not query.strip():
            return []
        hits = self.searcher.search(query, k)
        return [str(hit.docid) for hit in hits]

    def dense_search(self, query, k=FINAL_K):
        if not query or not query.strip():
            return []

        q_emb = self.bi_encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        _, indices = self.faiss_index.search(q_emb, k)
        return [self.doc_ids[i] for i in indices[0] if i != -1]

    @staticmethod
    def rrf_fusion(bm25_results, dense_results, k=RRF_K):
        scores = {}

        for rank, doc_id in enumerate(bm25_results):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

        for rank, doc_id in enumerate(dense_results):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in fused]

    def rerank(self, query, candidate_doc_ids, top_k=FINAL_K):
        if self.cross_encoder is None:
            raise RuntimeError("Cross-encoder was not loaded. Use load_reranker=True.")

        valid_doc_ids = [doc_id for doc_id in candidate_doc_ids if doc_id in self.documents]
        if not valid_doc_ids:
            return []

        pairs = [(query, self.documents[doc_id]) for doc_id in valid_doc_ids]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(valid_doc_ids, scores), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in ranked[:top_k]]

    def hybrid_search(self, query, k=FINAL_K, candidate_k=BM25_CANDIDATE_K):
        bm25_results = self.bm25_search(query, k=candidate_k)
        dense_results = self.dense_search(query, k=candidate_k)
        fused = self.rrf_fusion(bm25_results, dense_results, k=RRF_K)
        return fused[:k]

    def full_pipeline(self, query, k=FINAL_K, candidate_k=BM25_CANDIDATE_K, rerank_k=RERANK_K):
        bm25_results = self.bm25_search(query, k=candidate_k)
        dense_results = self.dense_search(query, k=candidate_k)
        fused = self.rrf_fusion(bm25_results, dense_results, k=RRF_K)
        rerank_candidates = fused[:rerank_k]
        return self.rerank(query, rerank_candidates, top_k=k)


def main():
    pipeline = RetrievalPipeline(load_reranker=True)
    query = "python list index error"

    print("Query:", query)
    print("BM25:", pipeline.bm25_search(query, k=5))
    print("Dense:", pipeline.dense_search(query, k=5))
    print("Hybrid:", pipeline.hybrid_search(query, k=5))
    print("Hybrid + Rerank:", pipeline.full_pipeline(query, k=5))


if __name__ == "__main__":
    main()
