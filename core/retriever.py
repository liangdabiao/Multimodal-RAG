import logging
from dataclasses import dataclass
from core.embedder import CohereEmbedder
from core.vector_store import VectorStore
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    doc_name: str
    page_idx: int
    score: float


class Retriever:
    def __init__(self, embedder: CohereEmbedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, doc_name: str = None,
                 top_k: int = None) -> list[RetrievalResult]:
        """Encode query, search Zilliz, return top-K most relevant pages."""
        if top_k is None:
            top_k = settings.top_k

        query_vector = self.embedder.encode_query(query)
        logger.info("Searching Zilliz, top_k=%d, doc_filter=%s", top_k, doc_name)
        hits = self.vector_store.search(query_vector, top_k=top_k, doc_name=doc_name)

        results = [
            RetrievalResult(
                doc_name=h["doc_name"],
                page_idx=h["page_idx"],
                score=round(h["score"], 4),
            )
            for h in hits
        ]
        logger.info("Search results: %s", [(r.doc_name, r.page_idx, r.score) for r in results])
        return results
