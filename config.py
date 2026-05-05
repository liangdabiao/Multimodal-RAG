import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # Embedding provider: "cohere" or "dashscope"
    embed_provider: str = "dashscope"

    # Cohere
    cohere_api_key: str = ""
    cohere_model: str = "embed-v4.0"
    cohere_dim: int = 1024
    cohere_batch_size: int = 96

    # DashScope
    dashscope_api_key: str = ""
    dashscope_model: str = "tongyi-embedding-vision-plus"
    dashscope_dim: int = 1152

    # Current active settings (resolved from provider)
    embed_model: str = ""
    embed_dim: int = 0

    # Milvus / Zilliz
    milvus_uri: str = ""
    milvus_token: str = ""
    collection_name: str = ""
    cohere_collection_name: str = "pdf_rag_cohere"
    dashscope_collection_name: str = "pdf_rag_dashscope"
    index_type: str = "IVF_FLAT"

    # Retrieval
    top_k: int = 3

    # LLM
    llm_provider: str = "dashscope"  # "openrouter" or "dashscope"
    openrouter_api_key: str = ""
    openrouter_model: str = "qwen/qwen3.5-397b-a17b"
    dashscope_vl_model: str = "qwen3.5-flash"
    generation_model: str = ""  # resolved from provider
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.7

    # PDF
    pdf_dpi: int = 150
    max_image_size: int = 1200

    def __post_init__(self):
        self._load_env()
        self._resolve_provider()

    def _load_env(self):
        self.embed_provider = os.getenv("EMBED_PROVIDER", self.embed_provider).lower()
        self.cohere_api_key = os.getenv("COHERE_API_KEY", self.cohere_api_key)
        self.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", self.dashscope_api_key)
        self.milvus_uri = os.getenv("MILVUS_HOST", self.milvus_uri)
        self.milvus_token = os.getenv("MILVUS_TOKEN", self.milvus_token)
        self.index_type = os.getenv("INDEX", self.index_type)
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", self.openrouter_api_key)
        self.cohere_collection_name = os.getenv("COHERE_COLLECTION_NAME", self.cohere_collection_name)
        self.dashscope_collection_name = os.getenv("DASHSCOPE_COLLECTION_NAME", self.dashscope_collection_name)
        self.llm_provider = os.getenv("LLM_PROVIDER", self.llm_provider).lower()
        self.dashscope_vl_model = os.getenv("DASHSCOPE_VL_MODEL", self.dashscope_vl_model)

    def _resolve_provider(self):
        if self.embed_provider == "cohere":
            self.embed_model = self.cohere_model
            self.embed_dim = self.cohere_dim
            self.collection_name = self.cohere_collection_name
        else:
            self.embed_model = self.dashscope_model
            self.embed_dim = self.dashscope_dim
            self.collection_name = self.dashscope_collection_name
        if self.llm_provider == "openrouter":
            self.generation_model = self.openrouter_model
        else:
            self.generation_model = self.dashscope_vl_model


settings = Settings()
