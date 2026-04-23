from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
DATA_DIR = APP_DIR / "data"
STATIC_DIR = APP_DIR / "static"
ASSETS_DIR = STATIC_DIR / "assets"

DOCUMENTS_PATH = DATA_DIR / "documents.pkl"
QUERY_DICT_PATH = DATA_DIR / "query_dict.pkl"
QRELS_DICT_PATH = DATA_DIR / "qrels_dict.pkl"

BM25_INDEX_DIR = DATA_DIR / "bm25_index"
DENSE_INDEX_PATH = DATA_DIR / "dense_index" / "stackoverflow_faiss.index"
DENSE_DOC_IDS_PATH = DATA_DIR / "dense_index" / "doc_ids.pkl"

JAVA_HOME_DEFAULT = "/usr/lib/jvm/java-11-openjdk-amd64"

BM25_CANDIDATE_K = 100
DENSE_CANDIDATE_K = 100
RRF_K = 60
RERANK_K = 20

DEFAULT_TOP_K = 10
MAX_TOP_K = 20

VALID_SYSTEMS = {
    "BM25",
    "Dense",
    "Hybrid (RRF)",
    "Hybrid + Rerank"
}
