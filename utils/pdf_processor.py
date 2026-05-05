import fitz  # PyMuPDF
from PIL import Image
from config import settings


class PdfProcessingError(Exception):
    pass


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
