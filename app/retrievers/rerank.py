from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name, device="cpu")

    def rerank(self, query: str, candidate_doc_ids, documents: dict, top_k: int = 10):
        if not candidate_doc_ids:
            return []

        valid_doc_ids = [doc_id for doc_id in candidate_doc_ids if doc_id in documents]
        if not valid_doc_ids:
            return []

        pairs = [(query, documents[doc_id]) for doc_id in valid_doc_ids]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(valid_doc_ids, scores), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in ranked[:top_k]]
