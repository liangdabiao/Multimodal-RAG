import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # Cohere Embedding API
    cohere_api_key: str = ""
    embed_model: str = "embed-v4.0"
    embed_dim: int = 1024
    cohere_batch_size: int = 96

    # Milvus / Zilliz
    milvus_uri: str = ""
    milvus_token: str = ""
    collection_name: str = "multimodal_search"
    index_type: str = "IVF_FLAT"

    # Retrieval
    top_k: int = 3

    # LLM (OpenRouter)
    openrouter_api_key: str = ""
    generation_model: str = "qwen/qwen3.5-397b-a17b"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.7

    # PDF
    pdf_dpi: int = 150
    max_image_size: int = 1200

    def __post_init__(self):
        self._load_env()

    def _load_env(self):
        self.cohere_api_key = os.getenv("COHERE_API_KEY", self.cohere_api_key)
        self.milvus_uri = os.getenv("MILVUS_HOST", self.milvus_uri)
        self.milvus_token = os.getenv("MILVUS_TOKEN", self.milvus_token)
        self.collection_name = os.getenv("COLLECTION_NAME", self.collection_name)
        self.index_type = os.getenv("INDEX", self.index_type)
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", self.openrouter_api_key)


settings = Settings()
