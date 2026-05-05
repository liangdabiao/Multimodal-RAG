import logging
from openai import OpenAI
from PIL import Image
from utils.image_utils import image_to_data_uri
from config import settings

logger = logging.getLogger(__name__)


class AnswerGenerator:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=120.0,
        )

    def generate(self, question: str, context_images: list[Image.Image]) -> str:
        """Send retrieved page images + question to Qwen3.5, return answer."""
        content = []
        for img in context_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_to_data_uri(img)},
            })
        content.append({
            "type": "text",
            "text": (
                f"Above are {len(context_images)} retrieved document pages.\n"
                f"Read them carefully and answer the following question:\n\n"
                f"Question: {question}\n\n"
                f"Be concise and accurate. If the documents don't contain "
                f"relevant information, say so."
            ),
        })

        logger.info(f"Calling LLM: {settings.generation_model}, {len(context_images)} images")
        response = self.client.chat.completions.create(
            model=settings.generation_model,
            messages=[{"role": "user", "content": content}],
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )

        text = response.choices[0].message.content
        if not text:
            logger.warning("LLM returned empty content. Model: %s, finish_reason: %s",
                           settings.generation_model,
                           response.choices[0].finish_reason)
            return "(LLM returned no content)"

        return text.strip()
