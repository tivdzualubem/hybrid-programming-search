import json
import os
import pickle
import shutil
import subprocess
import sys

import faiss
from sentence_transformers import SentenceTransformer

from config import (
    DOCUMENTS_PATH,
    BM25_CORPUS_DIR,
    BM25_INDEX_DIR,
    DENSE_INDEX_DIR,
    DOC_IDS_PATH,
    FAISS_INDEX_PATH,
    BI_ENCODER_MODEL,
)


def build_bm25_index(documents):
    os.makedirs(BM25_CORPUS_DIR, exist_ok=True)

    corpus_path = os.path.join(BM25_CORPUS_DIR, "corpus.jsonl")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc_id, text in documents.items():
            f.write(json.dumps({"id": doc_id, "contents": text}, ensure_ascii=False) + "\n")

    if os.path.exists(BM25_INDEX_DIR):
        shutil.rmtree(BM25_INDEX_DIR)

    command = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        BM25_CORPUS_DIR,
        "--index",
        BM25_INDEX_DIR,
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        "2",
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]

    print("Building Pyserini BM25 index...")
    subprocess.run(command, check=True)
    print("BM25 index saved to:", BM25_INDEX_DIR)


def build_faiss_index(documents):
    os.makedirs(DENSE_INDEX_DIR, exist_ok=True)

    doc_ids = list(documents.keys())
    doc_texts = [documents[doc_id] for doc_id in doc_ids]

    print("Loading bi-encoder:", BI_ENCODER_MODEL)
    bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)

    print("Encoding documents for dense index...")
    doc_embeddings = bi_encoder.encode(
        doc_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(DOC_IDS_PATH, "wb") as f:
        pickle.dump(doc_ids, f)

    print("FAISS index saved to:", FAISS_INDEX_PATH)
    print("Dense doc IDs saved to:", DOC_IDS_PATH)


def main():
    if not os.path.exists(DOCUMENTS_PATH):
        raise FileNotFoundError(
            "Missing artifacts/documents.pkl. Run: python scripts/export_artifacts.py"
        )

    with open(DOCUMENTS_PATH, "rb") as f:
        documents = pickle.load(f)

    build_bm25_index(documents)
    build_faiss_index(documents)


if __name__ == "__main__":
    main()
