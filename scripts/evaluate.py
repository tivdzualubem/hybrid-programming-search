import json
import math
import os
import pickle
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from config import (
    QUERY_DICT_PATH,
    QRELS_DICT_PATH,
    RESULTS_DIR,
    COMMON_RANK_DEPTH,
    BM25_CANDIDATE_K,
    RERANK_K,
)
from retrieval_pipeline import RetrievalPipeline


def precision_at_k(pred, true, k=10):
    pred_k = pred[:k]
    true_set = set(true)
    return len(set(pred_k) & true_set) / k if k else 0.0


def recall_at_k(pred, true, k=10):
    pred_k = pred[:k]
    true_set = set(true)
    if not true_set:
        return 0.0
    return len(set(pred_k) & true_set) / len(true_set)


def average_precision(pred, true):
    true_set = set(true)
    if not true_set:
        return 0.0

    hits = 0
    ap_sum = 0.0

    for i, doc_id in enumerate(pred, start=1):
        if doc_id in true_set:
            hits += 1
            ap_sum += hits / i

    return ap_sum / len(true_set)


def dcg_at_k(pred, true, k=10):
    true_set = set(true)
    dcg = 0.0

    for i, doc_id in enumerate(pred[:k], start=1):
        if doc_id in true_set:
            dcg += 1 / math.log2(i + 1)

    return dcg


def ndcg_at_k(pred, true, k=10):
    ideal_rels = min(len(set(true)), k)
    if ideal_rels == 0:
        return 0.0

    ideal_dcg = sum(1 / math.log2(i + 1) for i in range(1, ideal_rels + 1))
    return dcg_at_k(pred, true, k) / ideal_dcg if ideal_dcg else 0.0


def evaluate_system(model_func, name, query_dict, qrels_dict, query_limit=None, k=10):
    precisions = []
    recalls = []
    aps = []
    ndcgs = []

    items = list(query_dict.items())
    if query_limit is not None:
        items = items[:query_limit]

    for qid, query in items:
        true = qrels_dict.get(qid, [])
        if not true or not query.strip():
            continue

        pred = model_func(query)

        precisions.append(precision_at_k(pred, true, k=k))
        recalls.append(recall_at_k(pred, true, k=k))
        aps.append(average_precision(pred, true))
        ndcgs.append(ndcg_at_k(pred, true, k=k))

    return {
        "System": name,
        "Queries Evaluated": len(precisions),
        "Precision@10": float(np.mean(precisions)) if precisions else 0.0,
        "Recall@10": float(np.mean(recalls)) if recalls else 0.0,
        "MAP": float(np.mean(aps)) if aps else 0.0,
        "nDCG@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


def collect_per_query_ndcg(model_func, query_dict, qrels_dict, k=10):
    scores = []
    qids = []

    for qid, query in query_dict.items():
        true = qrels_dict.get(qid, [])
        if not true or not query.strip():
            continue

        pred = model_func(query)
        scores.append(ndcg_at_k(pred, true, k=k))
        qids.append(qid)

    return qids, np.array(scores)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(QUERY_DICT_PATH, "rb") as f:
        query_dict = pickle.load(f)

    with open(QRELS_DICT_PATH, "rb") as f:
        qrels_dict = pickle.load(f)

    pipeline = RetrievalPipeline(load_reranker=True)

    print("Evaluating systems...")

    bm25_metrics = evaluate_system(
        lambda q: pipeline.bm25_search(q, k=COMMON_RANK_DEPTH),
        "BM25",
        query_dict,
        qrels_dict,
        k=10,
    )

    dense_metrics = evaluate_system(
        lambda q: pipeline.dense_search(q, k=COMMON_RANK_DEPTH),
        "Dense",
        query_dict,
        qrels_dict,
        k=10,
    )

    hybrid_metrics = evaluate_system(
        lambda q: pipeline.hybrid_search(q, k=COMMON_RANK_DEPTH, candidate_k=BM25_CANDIDATE_K),
        "Hybrid (RRF)",
        query_dict,
        qrels_dict,
        k=10,
    )

    rerank_metrics = evaluate_system(
        lambda q: pipeline.full_pipeline(
            q,
            k=COMMON_RANK_DEPTH,
            candidate_k=BM25_CANDIDATE_K,
            rerank_k=RERANK_K,
        ),
        "Hybrid + Rerank (top-20, BEST)",
        query_dict,
        qrels_dict,
        k=10,
    )

    results_df = pd.DataFrame(
        [bm25_metrics, dense_metrics, hybrid_metrics, rerank_metrics]
    )

    print("\n=== FINAL RESULTS TABLE ===")
    print(results_df.to_string(index=False))

    results_path = os.path.join(RESULTS_DIR, "evaluation_results_final.csv")
    results_df.to_csv(results_path, index=False)
    print("Saved:", results_path)

    print("\nRunning Wilcoxon signed-rank tests on per-query nDCG@10...")
    _, scores_bm25 = collect_per_query_ndcg(
        lambda q: pipeline.bm25_search(q, k=100), query_dict, qrels_dict
    )
    _, scores_hybrid = collect_per_query_ndcg(
        lambda q: pipeline.hybrid_search(q, k=100, candidate_k=100), query_dict, qrels_dict
    )
    _, scores_rerank = collect_per_query_ndcg(
        lambda q: pipeline.full_pipeline(q, k=100, candidate_k=100, rerank_k=20),
        query_dict,
        qrels_dict,
    )

    stat_h, p_h = wilcoxon(scores_hybrid, scores_bm25, alternative="greater")
    stat_r, p_r = wilcoxon(scores_rerank, scores_bm25, alternative="greater")
    stat_hr, p_hr = wilcoxon(scores_hybrid, scores_rerank, alternative="greater")

    sig_results = {
        "Hybrid_RRF_vs_BM25_stat": float(stat_h),
        "Hybrid_RRF_vs_BM25_p": float(p_h),
        "Hybrid_Rerank_vs_BM25_stat": float(stat_r),
        "Hybrid_Rerank_vs_BM25_p": float(p_r),
        "Hybrid_RRF_vs_Hybrid_Rerank_stat": float(stat_hr),
        "Hybrid_RRF_vs_Hybrid_Rerank_p": float(p_hr),
    }

    sig_path = os.path.join(RESULTS_DIR, "significance_tests.json")
    with open(sig_path, "w") as f:
        json.dump(sig_results, f, indent=2)

    print("Saved:", sig_path)


if __name__ == "__main__":
    main()
