import io
import base64
from PIL import Image
from config import settings


def image_to_data_uri(img: Image.Image, max_size: int = None, fmt: str = "png") -> str:
    """Resize image so max dimension <= max_size, convert to base64 data URI."""
    if max_size is None:
        max_size = settings.max_image_size
    img = img.copy()
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        r = max_size / max(w, h)
        img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
    buf = io.BytesIO()
    if fmt == "jpeg":
        img.save(buf, format="JPEG", quality=85)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG")
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:{mime};base64,{b64}"


def pil_to_base64(img: Image.Image, max_size: int = None, fmt: str = "jpeg") -> str:
    """Resize image so max dimension <= max_size, return raw base64 string (no data URI prefix)."""
    if max_size is None:
        max_size = settings.max_image_size
    img = img.copy()
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        r = max_size / max(w, h)
        img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
    buf = io.BytesIO()
    if fmt == "jpeg":
        img.save(buf, format="JPEG", quality=85)
    else:
        img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
