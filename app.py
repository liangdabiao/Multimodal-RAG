import io
import logging
import os
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from config import settings
from utils.pdf_processor import pdf_to_images, PdfProcessingError
from core.vector_store import VectorStore
from core.embedder import create_embedder
from core.retriever import Retriever, expand_pages
from core.generator import AnswerGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

UPLOAD_DIR = Path(__file__).parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_components = {}
# image cache: doc_name -> list[PIL.Image], loaded on demand from PDF
_image_cache = {}


def get_components():
    if "vector_store" not in _components:
        logger.info("Connecting to Zilliz: %s", settings.milvus_uri)
        _components["vector_store"] = VectorStore()
    if "embedder" not in _components:
        logger.info("Initializing embedder: %s (%s)", settings.embed_model, settings.embed_provider)
        _components["embedder"] = create_embedder()
    if "retriever" not in _components:
        _components["retriever"] = Retriever(
            _components["embedder"], _components["vector_store"]
        )
    if "generator" not in _components:
        logger.info("Initializing LLM generator: %s", settings.generation_model)
        _components["generator"] = AnswerGenerator()
    return _components


def get_doc_names():
    """List all PDF files in the upload directory."""
    return sorted(f.name for f in UPLOAD_DIR.glob("*.pdf"))


def get_page_image(doc_name: str, page_idx: int):
    """Get a page image from cache or load from PDF on demand."""
    if doc_name not in _image_cache:
        pdf_path = UPLOAD_DIR / doc_name
        if pdf_path.exists():
            logger.info("Loading images from %s (on demand)", doc_name)
            _image_cache[doc_name] = pdf_to_images(str(pdf_path))
        else:
            return None
    images = _image_cache.get(doc_name, [])
    if page_idx < len(images):
        return images[page_idx]
    return None


@app.route("/")
def index():
    return send_file("static/index.html")


@app.route("/api/docs", methods=["GET"])
def list_docs():
    """Return all available documents (from disk, not just in-memory)."""
    return jsonify({"docs": get_doc_names()})


@app.route("/api/image/<doc_name>/<int:page_idx>")
def serve_image(doc_name, page_idx):
    """Serve a PDF page as PNG image."""
    img = get_page_image(doc_name, page_idx)
    if img is None:
        return "Not found", 404
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    doc_name = f.filename
    save_path = UPLOAD_DIR / doc_name
    f.save(str(save_path))
    logger.info("Saved: %s", save_path)

    # Invalidate cache if re-uploading same doc
    if doc_name in _image_cache:
        del _image_cache[doc_name]

    return jsonify({"doc_name": doc_name, "docs": get_doc_names()})


@app.route("/api/encode", methods=["POST"])
def encode():
    data = request.json or {}
    doc_name = data.get("doc_name", "")

    save_path = UPLOAD_DIR / doc_name
    if not save_path.exists():
        return jsonify({"error": f"File not found: {doc_name}"}), 400

    try:
        logger.info("Converting PDF to images: %s", doc_name)
        images = pdf_to_images(str(save_path))
        n_pages = len(images)
        _image_cache[doc_name] = images
        logger.info("Converted %d pages", n_pages)

        comps = get_components()
        logger.info("Encoding %d pages via %s API...", n_pages, settings.embed_provider)
        page_vectors = comps["embedder"].encode_images(images)
        logger.info("Got %d vectors", len(page_vectors))

        total_rows = comps["vector_store"].insert_pages(doc_name, page_vectors)
        logger.info("Inserted %d rows into Zilliz", total_rows)

        return jsonify({
            "status": "ok",
            "message": f"{doc_name}: {n_pages} pages indexed, {total_rows} vectors stored.",
        })
    except Exception as e:
        logger.error("Encode failed: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/search", methods=["POST"])
def search():
    data = request.json or {}
    question = data.get("question", "").strip()
    doc_name = data.get("doc_name", "")

    if not question:
        return jsonify({"error": "Please enter a question"}), 400

    # Check that at least some docs exist on disk
    if not get_doc_names():
        return jsonify({"error": "No documents uploaded yet"}), 400

    try:
        logger.info("Search: question='%s', doc='%s'", question[:80], doc_name)
        comps = get_components()
        filter_doc = None if doc_name == "__all__" else doc_name

        results = comps["retriever"].retrieve(
            question, doc_name=filter_doc, top_k=settings.top_k
        )
        logger.info("Retrieved %d pages", len(results))

        if not results:
            return jsonify({"pages": [], "answer": "未找到相关页面。"})

        expanded = expand_pages(results, UPLOAD_DIR, window=2)

        gallery = []
        context_images = []
        for doc_name_e, page_idx, score in expanded:
            img = get_page_image(doc_name_e, page_idx)
            if img:
                label = f"{doc_name_e} - 第{page_idx + 1}页 (相似度: {score})"
                gallery.append({"label": label, "doc_name": doc_name_e, "page_idx": page_idx})
                context_images.append(img)

        if not context_images:
            return jsonify({"pages": [], "answer": "源 PDF 文件未找到，请重新上传。"})

        logger.info("Sending %d images to LLM", len(context_images))
        answer = comps["generator"].generate(question, context_images)
        logger.info("LLM answer: %s", answer[:100])

        return jsonify({"pages": gallery, "answer": answer})
    except Exception as e:
        logger.error("Search failed: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/clear", methods=["POST"])
def clear():
    # Clear Zilliz
    try:
        comps = get_components()
        comps["vector_store"].drop_collection()
        comps["vector_store"]._ensure_collection(settings.collection_name)
    except Exception:
        pass

    # Clear disk files and cache
    for f in UPLOAD_DIR.glob("*.pdf"):
        f.unlink()
    _image_cache.clear()
    logger.info("All data cleared")
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    logger.info("Starting server on http://127.0.0.1:7860")
    app.run(host="127.0.0.1", port=7860, debug=False)
