from pyserini.search.lucene import LuceneSearcher


class BM25Retriever:
    def __init__(self, index_dir: str):
        self.searcher = LuceneSearcher(index_dir)

    def search(self, query: str, k: int = 10):
        if not query or not query.strip():
            return []
        hits = self.searcher.search(query, k)
        return [str(hit.docid) for hit in hits]
