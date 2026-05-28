import json
import os
import pickle
import platform
import random
import sys

import numpy as np
import pandas as pd
from datasets import load_dataset
import datasets
import sentence_transformers

from config import (
    SEED,
    CORPUS_SPLIT,
    QUERY_SPLIT,
    QRELS_SPLIT,
    ARTIFACTS_DIR,
    DOCUMENTS_PATH,
    QUERY_DICT_PATH,
    QRELS_DICT_PATH,
    BI_ENCODER_MODEL,
    CROSS_ENCODER_MODEL,
    BM25_CANDIDATE_K,
    DENSE_CANDIDATE_K,
    RRF_K,
    RERANK_K,
    FINAL_K,
    COMMON_RANK_DEPTH,
)

random.seed(SEED)
np.random.seed(SEED)

TARGET_LANGUAGES = {"python", "java", "javascript"}


def safe_join_text(title, text):
    title = title if title is not None else ""
    text = text if text is not None else ""
    return (title + " " + text).strip()


def try_filter_by_language(dataset, allowed_languages):
    if len(dataset) == 0:
        return dataset, False, "Empty dataset"

    sample = dataset[0]
    possible_keys = ["language", "lang", "tags", "tag", "topic", "topics"]
    found_key = None

    for key in possible_keys:
        if key in sample:
            found_key = key
            break

    if found_key is None:
        return dataset, False, "No language-like field found"

    def keep_item(item):
        value = item.get(found_key)
        if value is None:
            return False

        if isinstance(value, str):
            return value.strip().lower() in allowed_languages

        if isinstance(value, list):
            lowered = {str(v).strip().lower() for v in value}
            return len(lowered & allowed_languages) > 0

        return False

    filtered = dataset.filter(keep_item)
    return filtered, True, f"Filtered using field: {found_key}"


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print("Loading MTEB StackOverflowDupQuestions dataset...")
    corpus = load_dataset("mteb/stackoverflowdupquestions", "corpus")
    queries = load_dataset("mteb/stackoverflowdupquestions", "queries")
    qrels = load_dataset("mteb/stackoverflowdupquestions", "top_ranked")

    corpus_data = corpus[CORPUS_SPLIT]
    queries_data = queries[QUERY_SPLIT]
    qrels_data = qrels[QRELS_SPLIT]

    filtered_corpus_data, corpus_filtered, corpus_msg = try_filter_by_language(
        corpus_data, TARGET_LANGUAGES
    )
    filtered_queries_data, queries_filtered, queries_msg = try_filter_by_language(
        queries_data, TARGET_LANGUAGES
    )

    print("Corpus filter status:", corpus_msg)
    print("Queries filter status:", queries_msg)

    if corpus_filtered and len(filtered_corpus_data) > 0:
        corpus_data = filtered_corpus_data
    else:
        print("Using full corpus:", len(corpus_data))

    if queries_filtered and len(filtered_queries_data) > 0:
        queries_data = filtered_queries_data
    else:
        print("Using full queries:", len(queries_data))

    documents = {}
    for item in corpus_data:
        doc_id = str(item["_id"])
        documents[doc_id] = safe_join_text(item.get("title", ""), item.get("text", ""))

    query_dict = {}
    for item in queries_data:
        qid = str(item["_id"])
        text = item["text"].strip() if item["text"] is not None else ""
        query_dict[qid] = text

    qrels_dict = {}
    for item in qrels_data:
        qid = str(item["query-id"])
        rel_docs = [str(x) for x in item["corpus-ids"]]
        qrels_dict[qid] = rel_docs

    query_dict = {
        qid: qtext
        for qid, qtext in query_dict.items()
        if qid in qrels_dict and qtext.strip()
    }

    with open(DOCUMENTS_PATH, "wb") as f:
        pickle.dump(documents, f)

    with open(QUERY_DICT_PATH, "wb") as f:
        pickle.dump(query_dict, f)

    with open(QRELS_DICT_PATH, "wb") as f:
        pickle.dump(qrels_dict, f)

    run_config = {
        "seed": SEED,
        "corpus_split": CORPUS_SPLIT,
        "query_split": QUERY_SPLIT,
        "qrels_split": QRELS_SPLIT,
        "bi_encoder_model": BI_ENCODER_MODEL,
        "cross_encoder_model": CROSS_ENCODER_MODEL,
        "bm25_candidate_k": BM25_CANDIDATE_K,
        "dense_candidate_k": DENSE_CANDIDATE_K,
        "rrf_k": RRF_K,
        "rerank_k": RERANK_K,
        "final_k": FINAL_K,
        "common_rank_depth": COMMON_RANK_DEPTH,
    }

    env_info = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "datasets": datasets.__version__,
        "sentence_transformers": sentence_transformers.__version__,
    }

    with open(os.path.join(ARTIFACTS_DIR, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    with open(os.path.join(ARTIFACTS_DIR, "environment_info.json"), "w") as f:
        json.dump(env_info, f, indent=2)

    print("Saved preprocessing artifacts.")
    print("Documents:", len(documents))
    print("Queries:", len(query_dict))
    print("Qrels:", len(qrels_dict))


if __name__ == "__main__":
    main()
