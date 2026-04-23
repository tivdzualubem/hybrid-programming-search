import pickle
import faiss
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(self, index_path: str, doc_ids_path: str, model_name: str = "all-MiniLM-L6-v2"):
        self.index = faiss.read_index(index_path)
        with open(doc_ids_path, "rb") as f:
            self.doc_ids = pickle.load(f)
        self.encoder = SentenceTransformer(model_name, device="cpu")

    def search(self, query: str, k: int = 10):
        if not query or not query.strip():
            return []

        q_emb = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(q_emb, k)
        return [self.doc_ids[i] for i in indices[0] if i != -1]
