import os

SEED = 42

CORPUS_SPLIT = "test"
QUERY_SPLIT = "test"
QRELS_SPLIT = "test"

BI_ENCODER_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

BM25_CANDIDATE_K = 100
DENSE_CANDIDATE_K = 100
RRF_K = 60
RERANK_K = 20
FINAL_K = 10
COMMON_RANK_DEPTH = 20

ARTIFACTS_DIR = "artifacts"
BM25_CORPUS_DIR = os.path.join(ARTIFACTS_DIR, "bm25_corpus")
BM25_INDEX_DIR = os.path.join(ARTIFACTS_DIR, "bm25_index")
DENSE_INDEX_DIR = os.path.join(ARTIFACTS_DIR, "dense_index")
RESULTS_DIR = "results"

DOCUMENTS_PATH = os.path.join(ARTIFACTS_DIR, "documents.pkl")
QUERY_DICT_PATH = os.path.join(ARTIFACTS_DIR, "query_dict.pkl")
QRELS_DICT_PATH = os.path.join(ARTIFACTS_DIR, "qrels_dict.pkl")
DOC_IDS_PATH = os.path.join(DENSE_INDEX_DIR, "doc_ids.pkl")
FAISS_INDEX_PATH = os.path.join(DENSE_INDEX_DIR, "stackoverflow_faiss.index")
