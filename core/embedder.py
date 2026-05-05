import io
import base64
import logging
import time
from PIL import Image
from config import settings
from utils.image_utils import pil_to_base64

logger = logging.getLogger(__name__)

# Rate limits
_WINDOW = 60

# Cohere free tier
COHERE_IMAGE_LIMIT = 5
COHERE_TEXT_LIMIT = 2000


class _RateLimiter:
    def __init__(self, limit: int, window: int = _WINDOW):
        self.limit = limit
        self.window = window
        self.timestamps: list[float] = []

    def wait(self, count: int = 1):
        now = time.time()
        self.timestamps = [t for t in self.timestamps if now - t < self.window]
        remaining = self.limit - len(self.timestamps)
        if count > remaining:
            sleep_time = self.window - (now - self.timestamps[0]) + 0.5
            if sleep_time > 0:
                logger.warning("Rate limit: need %d slots but %d remaining, sleeping %.0fs",
                               count, remaining, sleep_time)
                time.sleep(sleep_time)
        for _ in range(count):
            self.timestamps.append(time.time())


class BaseEmbedder:
    """Common interface for embedding providers."""

    def encode_images(self, images: list[Image.Image]) -> list[list[float]]:
        raise NotImplementedError

    def encode_query(self, query: str) -> list[float]:
        raise NotImplementedError


class CohereEmbedder(BaseEmbedder):
    """Cohere embed-v4.0 via Cohere API."""

    def __init__(self):
        import cohere
        self.client = cohere.ClientV2(api_key=settings.cohere_api_key)
        self.image_limiter = _RateLimiter(COHERE_IMAGE_LIMIT)
        self.text_limiter = _RateLimiter(COHREE_TEXT_LIMIT)

    def encode_images(self, images: list[Image.Image]) -> list[list[float]]:
        batch_size = settings.cohere_batch_size
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            n = len(batch)

            self.image_limiter.wait(n)
            logger.info("[Cohere] Encoding batch %d-%d/%d (%d images)...",
                        i, min(i + batch_size, len(images)), len(images), n)

            from utils.image_utils import image_to_data_uri
            data_uris = [image_to_data_uri(img, max_size=settings.max_image_size, fmt="jpeg") for img in batch]

            response = self.client.embed(
                images=data_uris,
                model=settings.embed_model,
                input_type="search_document",
                embedding_types=["float"],
                output_dimension=settings.embed_dim,
            )
            batch_embs = response.embeddings.float_
            logger.info("[Cohere] Got %d vectors (dim=%d)", len(batch_embs), len(batch_embs[0]) if batch_embs else 0)
            all_embeddings.extend(batch_embs)

        return all_embeddings

    def encode_query(self, query: str) -> list[float]:
        self.text_limiter.wait(1)
        logger.info("[Cohere] Encoding query: %s", query[:50])
        response = self.client.embed(
            texts=[query],
            model=settings.embed_model,
            input_type="search_query",
            embedding_types=["float"],
            output_dimension=settings.embed_dim,
        )
        vec = response.embeddings.float_[0]
        logger.info("[Cohere] Query vector dim=%d", len(vec))
        return vec


class DashScopeEmbedder(BaseEmbedder):
    """tongyi-embedding-vision-plus via DashScope API."""

    def __init__(self):
        import dashscope
        dashscope.api_key = settings.dashscope_api_key

    def encode_images(self, images: list[Image.Image]) -> list[list[float]]:
        all_embeddings = []
        # tongyi-embedding-vision-plus: max 8 images per request
        batch_size = 8

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            n = len(batch)

            logger.info("[DashScope] Encoding batch %d-%d/%d (%d images)...",
                        i, min(i + batch_size, len(images)), len(images), n)

            input_data = [{"image": pil_to_base64(img, max_size=settings.max_image_size)} for img in batch]

            import dashscope
            from http import HTTPStatus
            resp = dashscope.MultiModalEmbedding.call(
                model=settings.embed_model,
                input=input_data,
            )
            if resp.status_code != HTTPStatus.OK:
                raise RuntimeError(f"DashScope API error: {resp.status_code} - {resp.message}")

            for item in resp.output["embeddings"]:
                all_embeddings.append(item["embedding"])
            logger.info("[DashScope] Got %d vectors (dim=%d)", len(all_embeddings),
                        len(all_embeddings[0]) if all_embeddings else 0)

        return all_embeddings

    def encode_query(self, query: str) -> list[float]:
        logger.info("[DashScope] Encoding query: %s", query[:50])

        import dashscope
        from http import HTTPStatus
        resp = dashscope.MultiModalEmbedding.call(
            model=settings.embed_model,
            input=[{"text": query}],
        )
        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(f"DashScope API error: {resp.status_code} - {resp.message}")

        vec = resp.output["embeddings"][0]["embedding"]
        logger.info("[DashScope] Query vector dim=%d", len(vec))
        return vec


def create_embedder() -> BaseEmbedder:
    """Factory: create embedder based on settings.embed_provider."""
    provider = settings.embed_provider.lower()
    if provider == "cohere":
        logger.info("Using Cohere embedder (model=%s, dim=%d)", settings.embed_model, settings.embed_dim)
        return CohereEmbedder()
    elif provider in ("dashscope", "qwen", "tongyi"):
        logger.info("Using DashScope embedder (model=%s, dim=%d)", settings.embed_model, settings.embed_dim)
        return DashScopeEmbedder()
    else:
        raise ValueError(f"Unknown embed provider: {provider}. Use 'cohere' or 'dashscope'.")
