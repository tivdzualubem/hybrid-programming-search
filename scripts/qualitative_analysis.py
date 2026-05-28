import os

from config import RESULTS_DIR
from retrieval_pipeline import RetrievalPipeline


SAMPLE_QUERIES = [
    "python list index error",
    "javascript referenceerror is not defined",
    "java null pointer exception in arraylist",
    "how to convert string to int in python",
    "website is white after running javascript",
]


def append_result_block(lines, system_name, doc_ids, documents, top_k=5):
    lines.append(f"\n--- {system_name} ---")
    for rank, doc_id in enumerate(doc_ids[:top_k], start=1):
        text = documents.get(doc_id, "")
        preview = text[:220].replace("\n", " ")
        lines.append(f"{rank}. DOC ID: {doc_id}")
        lines.append(preview)
        lines.append("")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    pipeline = RetrievalPipeline(load_reranker=True)
    lines = []

    for query in SAMPLE_QUERIES:
        lines.append("=" * 100)
        lines.append(f"QUERY: {query}")
        lines.append("=" * 100)

        bm25_res = pipeline.bm25_search(query, k=5)
        dense_res = pipeline.dense_search(query, k=5)
        hybrid_res = pipeline.hybrid_search(query, k=5)
        rerank_res = pipeline.full_pipeline(query, k=5)

        append_result_block(lines, "BM25", bm25_res, pipeline.documents)
        append_result_block(lines, "Dense", dense_res, pipeline.documents)
        append_result_block(lines, "Hybrid (RRF)", hybrid_res, pipeline.documents)
        append_result_block(lines, "Hybrid + Rerank (top-20)", rerank_res, pipeline.documents)

    output_path = os.path.join(RESULTS_DIR, "qualitative_analysis.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Saved:", output_path)


if __name__ == "__main__":
    main()
