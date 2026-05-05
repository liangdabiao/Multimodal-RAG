import logging
from dataclasses import dataclass
from pathlib import Path
from core.embedder import BaseEmbedder
from core.vector_store import VectorStore
from config import settings
from utils.pdf_processor import get_page_count

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    doc_name: str
    page_idx: int
    score: float


class Retriever:
    def __init__(self, embedder: BaseEmbedder, vector_store: VectorStore):
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


def expand_pages(results: list[RetrievalResult], upload_dir: Path,
                 window: int = 2) -> list[tuple[str, int, float]]:
    """Expand each hit ±window pages, deduplicate, stay within valid page range.

    Returns list of (doc_name, page_idx, score) tuples.
    """
    expanded: dict[tuple[str, int], float] = {}
    for r in results:
        pdf_path = upload_dir / r.doc_name
        total = get_page_count(str(pdf_path))
        for offset in range(-window, window + 1):
            idx = r.page_idx + offset
            if 0 <= idx < total:
                key = (r.doc_name, idx)
                if key not in expanded or r.score > expanded[key]:
                    expanded[key] = r.score

    items = [(doc, idx, score) for (doc, idx), score in expanded.items()]
    logger.info("Expanded %d hits ±%d → %d unique pages", len(results), window, len(items))
    return items
