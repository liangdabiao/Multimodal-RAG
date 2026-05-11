"""
独立 API 服务 —— 对外提供问答接口

启动：python api_server.py
端口：7861

POST /api/query
{
    "question": "问题内容",
    "doc_name": "xxx.pdf"   // 可选，不传则搜索全部文档
}

响应：
{
    "answer": "回答内容",
    "pages": [
        {"doc_name": "xxx.pdf", "page_idx": 0, "score": 0.82},
        ...
    ]
}
"""

import logging
import traceback
from pathlib import Path

from flask import Flask, request, jsonify
from PIL import Image
from config import settings
from utils.pdf_processor import pdf_page_to_image, pdf_pages_to_images
from core.embedder import create_embedder
from core.vector_store import VectorStore
from core.retriever import Retriever, expand_pages
from core.generator import AnswerGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

UPLOAD_DIR = Path(__file__).parent / "data" / "uploads"

_components = {}
_page_cache: dict[tuple[str, int], Image.Image] = {}
_PAGE_CACHE_MAX = 64


def _load_pages_batch(pages_needed: dict[str, list[int]]):
    for doc_name, indices in pages_needed.items():
        pdf_path = UPLOAD_DIR / doc_name
        if not pdf_path.exists():
            continue
        uncached = [i for i in indices if (doc_name, i) not in _page_cache]
        if not uncached:
            continue
        try:
            loaded = pdf_pages_to_images(str(pdf_path), uncached)
            for idx, img in loaded.items():
                _page_cache[(doc_name, idx)] = img
            _evict_cache()
        except Exception:
            logger.warning("[API] Batch load failed for %s", doc_name)


def _evict_cache():
    if len(_page_cache) > _PAGE_CACHE_MAX:
        keys = list(_page_cache.keys())
        for k in keys[: len(keys) - _PAGE_CACHE_MAX + 16]:
            del _page_cache[k]


def get_components():
    if "vector_store" not in _components:
        _components["vector_store"] = VectorStore()
    if "embedder" not in _components:
        _components["embedder"] = create_embedder()
    if "retriever" not in _components:
        _components["retriever"] = Retriever(
            _components["embedder"], _components["vector_store"]
        )
    if "generator" not in _components:
        _components["generator"] = AnswerGenerator()
    return _components


def get_doc_names():
    return sorted(f.name for f in UPLOAD_DIR.glob("*.pdf"))


@app.route("/api/query", methods=["POST"])
def query():
    data = request.json or {}
    question = data.get("question", "").strip()
    doc_name = data.get("doc_name", "")

    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    if not get_doc_names():
        return jsonify({"error": "没有已上传的文档"}), 400

    try:
        logger.info("[API] 查询: question='%s', doc='%s'", question[:80], doc_name)
        comps = get_components()
        filter_doc = None if not doc_name else doc_name

        results = comps["retriever"].retrieve(
            question, doc_name=filter_doc, top_k=settings.top_k
        )
        logger.info("[API] 检索到 %d 页", len(results))

        if not results:
            return jsonify({"answer": "未找到相关页面。", "pages": []})

        expanded = expand_pages(results, UPLOAD_DIR, window=2)

        # Batch-load all expanded pages
        pages_by_doc: dict[str, list[int]] = {}
        for doc_name_e, page_idx, _ in expanded:
            pages_by_doc.setdefault(doc_name_e, []).append(page_idx)
        _load_pages_batch(pages_by_doc)

        # Return all expanded pages info
        pages = []
        for doc_name_e, page_idx, score in expanded:
            if (doc_name_e, page_idx) in _page_cache:
                pages.append({
                    "doc_name": doc_name_e,
                    "page_idx": page_idx,
                    "score": score,
                })

        # LLM: only core hit pages
        core_images = []
        for r in results:
            key = (r.doc_name, r.page_idx)
            if key in _page_cache:
                core_images.append(_page_cache[key])

        if not core_images:
            return jsonify({"answer": "源 PDF 文件未找到，请重新上传。", "pages": pages})

        logger.info("[API] LLM: %d core images (expanded: %d pages)", len(core_images), len(pages))
        answer = comps["generator"].generate(question, core_images)
        logger.info("[API] 回答: %s", answer[:100])

        return jsonify({"answer": answer, "pages": pages})
    except Exception as e:
        logger.error("[API] 查询失败: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/docs", methods=["GET"])
def list_docs():
    return jsonify({"docs": get_doc_names()})


if __name__ == "__main__":
    logger.info("API 服务启动: http://127.0.0.1:7861")
    app.run(host="127.0.0.1", port=7861, debug=False)
