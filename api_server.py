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

import io
import logging
import traceback
from pathlib import Path

from flask import Flask, request, jsonify, send_file
from config import settings
from utils.pdf_processor import pdf_to_images
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
_image_cache = {}


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


def get_page_image(doc_name: str, page_idx: int):
    if doc_name not in _image_cache:
        pdf_path = UPLOAD_DIR / doc_name
        if pdf_path.exists():
            _image_cache[doc_name] = pdf_to_images(str(pdf_path))
        else:
            return None
    images = _image_cache.get(doc_name, [])
    return images[page_idx] if page_idx < len(images) else None


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

        pages = []
        context_images = []
        for doc_name_e, page_idx, score in expanded:
            img = get_page_image(doc_name_e, page_idx)
            if img:
                pages.append({
                    "doc_name": doc_name_e,
                    "page_idx": page_idx,
                    "score": score,
                })
                context_images.append(img)

        if not context_images:
            return jsonify({"answer": "源 PDF 文件未找到，请重新上传。", "pages": []})

        logger.info("[API] 发送 %d 张图片给 LLM", len(context_images))
        answer = comps["generator"].generate(question, context_images)
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
