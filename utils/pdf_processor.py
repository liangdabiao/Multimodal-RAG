import fitz  # PyMuPDF
from PIL import Image
from config import settings


class PdfProcessingError(Exception):
    pass


def get_page_count(pdf_path: str) -> int:
    """Return total page count without loading images."""
    try:
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return 0


def pdf_page_to_image(pdf_path: str, page_idx: int, dpi: int = None) -> Image.Image:
    """Load a single page from PDF as PIL Image."""
    if dpi is None:
        dpi = settings.pdf_dpi
    zoom = dpi / 72
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_idx]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    finally:
        doc.close()
    return img


def pdf_pages_to_images(pdf_path: str, page_indices: list[int], dpi: int = None) -> dict[int, Image.Image]:
    """Load multiple pages in one PDF open/close cycle. Returns {page_idx: Image}."""
    if not page_indices:
        return {}
    if dpi is None:
        dpi = settings.pdf_dpi
    zoom = dpi / 72
    result = {}
    doc = fitz.open(pdf_path)
    try:
        total = len(doc)
        for idx in page_indices:
            if 0 <= idx < total:
                page = doc[idx]
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                w = pix.width
                h = pix.height
                s = pix.samples
                result[idx] = Image.frombytes("RGB", [w, h], s)
    finally:
        doc.close()
    return result


def pdf_to_images(pdf_path: str, dpi: int = None) -> list[Image.Image]:
    """Convert each PDF page to RGB PIL Image using PyMuPDF (no poppler needed)."""
    if dpi is None:
        dpi = settings.pdf_dpi

    zoom = dpi / 72  # PDF default is 72 DPI
    try:
        doc = fitz.open(pdf_path)
        images = []
        for page in doc:
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        doc.close()
        return images
    except Exception as e:
        raise PdfProcessingError(f"Failed to process PDF: {e}")
