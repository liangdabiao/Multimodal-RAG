import logging
from openai import OpenAI
from PIL import Image
from utils.image_utils import image_to_data_uri
from config import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "以上是 {n} 个检索到的文档页面。"
    "请仔细阅读并回答以下问题。\n\n"
    "问题：{question}\n\n"
    "请用中文回答，简洁准确。如果文档中没有相关信息，请说明。"
)


class AnswerGenerator:
    def __init__(self):
        if settings.llm_provider == "openrouter":
            self.client = OpenAI(
                api_key=settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                timeout=120.0,
            )
        else:
            self.client = OpenAI(
                api_key=settings.dashscope_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                timeout=120.0,
            )

    def generate(self, question: str, context_images: list[Image.Image]) -> str:
        content = []
        for img in context_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_to_data_uri(img)},
            })
        content.append({
            "type": "text",
            "text": _SYSTEM_PROMPT.format(n=len(context_images), question=question),
        })

        logger.info("Calling LLM [%s]: %s, %d images",
                     settings.llm_provider, settings.generation_model, len(context_images))
        response = self.client.chat.completions.create(
            model=settings.generation_model,
            messages=[{"role": "user", "content": content}],
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )

        text = response.choices[0].message.content
        if not text:
            logger.warning("LLM returned empty content. Provider: %s, model: %s, finish_reason: %s",
                           settings.llm_provider, settings.generation_model,
                           response.choices[0].finish_reason)
            return "(LLM returned no content)"

        return text.strip()
