import os
import pickle
from typing import List

from app.config import (
    DOCUMENTS_PATH,
    BM25_INDEX_DIR,
    DENSE_INDEX_PATH,
    DENSE_DOC_IDS_PATH,
    JAVA_HOME_DEFAULT,
    BM25_CANDIDATE_K,
    DENSE_CANDIDATE_K,
    RRF_K,
    RERANK_K,
    DEFAULT_TOP_K,
    MAX_TOP_K,
)
from app.retrievers.bm25 import BM25Retriever
from app.retrievers.dense import DenseRetriever
from app.retrievers.fusion import rrf_fusion
from app.retrievers.rerank import CrossEncoderReranker


class SearchService:
    def __init__(self):
        os.environ.setdefault("JAVA_HOME", JAVA_HOME_DEFAULT)
        self.documents = self._load_pickle(DOCUMENTS_PATH)
        self.bm25 = None
        self.dense = None
        self.reranker = None

    @staticmethod
    def _load_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def _ensure_bm25(self):
        if self.bm25 is None:
            self.bm25 = BM25Retriever(str(BM25_INDEX_DIR))

    def _ensure_dense(self):
        if self.dense is None:
            self.dense = DenseRetriever(
                index_path=str(DENSE_INDEX_PATH),
                doc_ids_path=str(DENSE_DOC_IDS_PATH),
            )

    def _ensure_reranker(self):
        if self.reranker is None:
            self.reranker = CrossEncoderReranker()

    @staticmethod
    def _preview(text: str, max_len: int = 280) -> str:
        clean = " ".join(text.split())
        return clean[:max_len] + ("..." if len(clean) > max_len else "")

    def search(self, query: str, system: str, top_k: int = DEFAULT_TOP_K) -> List[dict]:
        top_k = max(1, min(int(top_k), MAX_TOP_K))

        if system == "BM25":
            self._ensure_bm25()
            doc_ids = self.bm25.search(query, k=top_k)

        elif system == "Dense":
            self._ensure_dense()
            doc_ids = self.dense.search(query, k=top_k)

        elif system == "Hybrid (RRF)":
            self._ensure_bm25()
            self._ensure_dense()
            bm25_ids = self.bm25.search(query, k=BM25_CANDIDATE_K)
            dense_ids = self.dense.search(query, k=DENSE_CANDIDATE_K)
            doc_ids = rrf_fusion(bm25_ids, dense_ids, k=RRF_K)[:top_k]

        elif system == "Hybrid + Rerank":
            self._ensure_bm25()
            self._ensure_dense()
            self._ensure_reranker()
            bm25_ids = self.bm25.search(query, k=BM25_CANDIDATE_K)
            dense_ids = self.dense.search(query, k=DENSE_CANDIDATE_K)
            fused_ids = rrf_fusion(bm25_ids, dense_ids, k=RRF_K)
            rerank_candidates = fused_ids[:RERANK_K]
            doc_ids = self.reranker.rerank(query, rerank_candidates, self.documents, top_k=top_k)

        else:
            raise ValueError(f"Unsupported system: {system}")

        results = []
        for rank, doc_id in enumerate(doc_ids, start=1):
            preview = self._preview(self.documents.get(doc_id, ""))
            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "preview": preview
            })
        return results


search_service = SearchService()
