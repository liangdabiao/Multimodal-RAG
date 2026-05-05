import io
import base64
import logging
import tempfile
import os
import traceback
from flask import Flask, request, jsonify, send_file
from config import settings
from utils.pdf_processor import pdf_to_images, PdfProcessingError
from core.vector_store import VectorStore
from core.embedder import CohereEmbedder
from core.retriever import Retriever
from core.generator import AnswerGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

_components = {}
_doc_state = {"docs": [], "images": {}, "paths": {}}


def get_components():
    if "vector_store" not in _components:
        logger.info("Connecting to Zilliz: %s", settings.milvus_uri)
        _components["vector_store"] = VectorStore()
    if "embedder" not in _components:
        logger.info("Initializing Cohere embedder: %s", settings.embed_model)
        _components["embedder"] = CohereEmbedder()
    if "retriever" not in _components:
        _components["retriever"] = Retriever(
            _components["embedder"], _components["vector_store"]
        )
    if "generator" not in _components:
        logger.info("Initializing LLM generator: %s", settings.generation_model)
        _components["generator"] = AnswerGenerator()
    return _components


def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.route("/")
def index():
    return send_file("static/index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    doc_name = f.filename
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    f.save(tmp.name)
    tmp.close()
    logger.info("Uploaded: %s -> %s", doc_name, tmp.name)

    _doc_state["paths"][doc_name] = tmp.name
    if doc_name not in _doc_state["docs"]:
        _doc_state["docs"].append(doc_name)

    return jsonify({"doc_name": doc_name, "docs": _doc_state["docs"]})


@app.route("/api/encode", methods=["POST"])
def encode():
    data = request.json or {}
    doc_name = data.get("doc_name", "")

    if doc_name not in _doc_state["paths"]:
        return jsonify({"error": "Please upload a PDF first"}), 400

    try:
        pdf_path = _doc_state["paths"][doc_name]
        logger.info("Converting PDF to images: %s", doc_name)
        images = pdf_to_images(pdf_path)
        _doc_state["images"][doc_name] = images
        n_pages = len(images)
        logger.info("Converted %d pages", n_pages)

        comps = get_components()
        logger.info("Encoding %d pages via Cohere API...", n_pages)
        page_vectors = comps["embedder"].encode_images(images)
        logger.info("Got %d vectors, dim=%d", len(page_vectors), len(page_vectors[0]) if page_vectors else 0)

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
    if not _doc_state["images"]:
        return jsonify({"error": "No documents indexed yet"}), 400

    try:
        logger.info("Search: question='%s', doc='%s'", question[:50], doc_name)
        comps = get_components()
        filter_doc = None if doc_name == "__all__" else doc_name

        results = comps["retriever"].retrieve(
            question, doc_name=filter_doc, top_k=settings.top_k
        )
        logger.info("Retrieved %d pages", len(results))

        if not results:
            return jsonify({"pages": [], "answer": "No relevant pages found."})

        gallery = []
        context_images = []
        for r in results:
            imgs = _doc_state["images"].get(r.doc_name, [])
            if r.page_idx < len(imgs):
                label = f"{r.doc_name} - Page {r.page_idx + 1} (score: {r.score})"
                b64 = pil_to_base64(imgs[r.page_idx])
                gallery.append({"label": label, "image": b64})
                context_images.append(imgs[r.page_idx])

        if not context_images:
            return jsonify({"pages": gallery, "answer": "Images not found in cache."})

        answer = comps["generator"].generate(question, context_images)
        logger.info("LLM answer: %s", answer[:100])

        return jsonify({"pages": gallery, "answer": answer})
    except Exception as e:
        logger.error("Search failed: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/clear", methods=["POST"])
def clear():
    _doc_state["docs"] = []
    _doc_state["images"] = {}
    for path in _doc_state["paths"].values():
        try:
            os.unlink(path)
        except OSError:
            pass
    _doc_state["paths"] = {}
    try:
        comps = get_components()
        comps["vector_store"].drop_collection()
        comps["vector_store"]._ensure_collection(settings.collection_name)
    except Exception:
        pass
    logger.info("All data cleared")
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    logger.info("Starting server on http://127.0.0.1:7860")
    app.run(host="127.0.0.1", port=7860, debug=False)
