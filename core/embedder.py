import logging
from PIL import Image
import cohere
from config import settings
from utils.image_utils import image_to_data_uri

logger = logging.getLogger(__name__)


class CohereEmbedder:
    def __init__(self):
        self.client = cohere.ClientV2(api_key=settings.cohere_api_key)

    def encode_images(self, images: list[Image.Image]) -> list[list[float]]:
        """Encode page images into vectors via Cohere API."""
        batch_size = settings.cohere_batch_size
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            logger.info("Encoding batch %d-%d/%d via Cohere...", i, min(i + batch_size, len(images)), len(images))

            data_uris = [image_to_data_uri(img, max_size=settings.max_image_size, fmt="jpeg") for img in batch]

            response = self.client.embed(
                images=data_uris,
                model=settings.embed_model,
                input_type="search_document",
                embedding_types=["float"],
                output_dimension=settings.embed_dim,
            )
            batch_embs = response.embeddings.float_
            logger.info("Got %d vectors from this batch", len(batch_embs))
            all_embeddings.extend(batch_embs)

        return all_embeddings

    def encode_query(self, query: str) -> list[float]:
        """Encode a text query into a single vector via Cohere API."""
        logger.info("Encoding query: %s", query[:50])
        response = self.client.embed(
            texts=[query],
            model=settings.embed_model,
            input_type="search_query",
            embedding_types=["float"],
            output_dimension=settings.embed_dim,
        )
        vec = response.embeddings.float_[0]
        logger.info("Query vector dim=%d", len(vec))
        return vec
