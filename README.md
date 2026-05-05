# Multimodal RAG - PDF Intelligent Q&A

基于 Cohere embed-v4.0 + Zilliz + Qwen3.5 的多模态 RAG 系统。上传 PDF，用自然语言提问，系统自动检索最相关的页面并由 AI 生成回答。

与传统 RAG 不同，本系统**不做文本提取和 OCR**，而是直接将 PDF 页面当作图片处理，通过视觉 Embedding 模型编码，完整保留表格、图表、排版、手写批注等所有视觉信息。

## 工作原理

```
PDF 文档
  │
  ▼ (PyMuPDF, 150 DPI)
页面图片（不做任何文本提取）
  │
  ▼ (Cohere embed-v4.0 API, 云端调用)
每页图片 → 1 个 1024 维向量
  │
  ▼ (写入 Zilliz Serverless 云向量数据库)
doc_name + page_idx + vector
  │
  ══════════════ 用户提问时 ══════════════
  │
  ▼ (Cohere embed-v4.0 API 编码查询文本)
查询 → 1 个 1024 维向量
  │
  ▼ (Zilliz 内积相似度搜索)
Top-K 最相关页面
  │
  ▼ (页面原始图片 + 问题 → Qwen3.5-397B-A17B 多模态大模型, via OpenRouter)
AI 直接"看图"回答问题
```

**核心优势**：
- 无需本地 GPU，无需安装 PyTorch 或 poppler
- 所有 AI 计算通过云端 API 完成（Cohere Embedding + OpenRouter LLM）
- 安装依赖只需几秒钟，8 个轻量 Python 包
- 面对扫描件 PDF、图文混排文档、含公式与表格的专业资料，都能完整保留所有视觉信息

## 技术栈

| 组件 | 技术 | 说明 |
|---|---|---|
| PDF → 图片 | PyMuPDF (fitz) | 纯 Python，无需 poppler，跨平台直接可用 |
| 图片/文本编码 | [Cohere embed-v4.0](https://docs.cohere.com/docs/embeddings) API | 云端多模态 Embedding，支持图片+文本，每页产出 1 个 1024 维向量 |
| 向量数据库 | [Zilliz Serverless](https://cloud.zilliz.com) (云 Milvus) | IVF_FLAT 索引，内积相似度，零运维 |
| 生成模型 | Qwen3.5-397B-A17B (via [OpenRouter](https://openrouter.ai)) | 多模态大模型，直接"看"页面图片生成回答 |
| Web 界面 | Flask + 原生 HTML/JS | 轻量无框架依赖，无 telemetry，本地运行 |

## 项目结构

```
D:/PDF-AI/
├── .env                    # 密钥和连接配置（不入库）
├── .env.example            # 配置模板
├── .gitignore
├── requirements.txt        # Python 依赖（8 个包，无 PyTorch）
├── config.py               # 配置中心，从 .env 加载
├── app.py                  # Flask Web 服务 + API 路由
├── static/
│   └── index.html          # 前端页面
│
├── core/
│   ├── embedder.py         # Cohere Embedding API 封装（图片编码 + 查询编码）
│   ├── vector_store.py     # Zilliz 向量库：集合管理、插入、搜索
│   ├── retriever.py        # 单向量检索：编码查询 → 搜索 Zilliz → Top-K 页面
│   └── generator.py        # LLM 回答生成：OpenRouter + Qwen3.5 多模态
│
└── utils/
    ├── pdf_processor.py    # PDF 转图片（PyMuPDF 封装，无需 poppler）
    └── image_utils.py      # 图片转 base64 data URI（供 Cohere 和 LLM 使用）
```

## API 接口

| 方法 | 路径 | 说明 | 请求体 |
|---|---|---|---|
| POST | `/api/upload` | 上传 PDF 文件 | `multipart/form-data`，字段 `file` |
| POST | `/api/encode` | 编码已上传的 PDF 并写入向量库 | `{"doc_name": "xxx.pdf"}` |
| POST | `/api/search` | 检索相关页面并生成回答 | `{"question": "...", "doc_name": "xxx.pdf"}` |
| POST | `/api/clear` | 清空所有数据 | 无 |

搜索接口返回示例：
```json
{
  "pages": [
    {"label": "report.pdf - Page 5 (score: 0.8234)", "image": "base64..."},
    {"label": "report.pdf - Page 12 (score: 0.7891)", "image": "base64..."}
  ],
  "answer": "根据文档内容，该系统的核心区别在于..."
}
```

## 快速开始

### 1. 环境要求

- Python 3.10+
- 无需 GPU、无需 poppler、无需 CUDA
- 网络能访问 Cohere API、OpenRouter API 和 Zilliz Cloud

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

共 8 个轻量包（cohere、pymilvus、openai、PyMuPDF、pillow、flask、numpy、python-dotenv），安装通常在 1 分钟内完成。

### 3. 获取 API Key

**Cohere**（Embedding 编码，必需）：前往 [https://dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys) 注册获取，有免费额度。

**OpenRouter**（LLM 生成，必需）：前往 [https://openrouter.ai/settings/keys](https://openrouter.ai/settings/keys) 注册获取。

**Zilliz Cloud**（向量数据库，必需）：前往 [https://cloud.zilliz.com](https://cloud.zilliz.com) 创建 Serverless 集群，获取连接地址和 Token。

### 4. 配置 .env

复制 `.env.example` 为 `.env`，填入实际值：

```env
# Cohere Embedding API Key（必需）
COHERE_API_KEY=your-cohere-api-key

# OpenRouter API Key（必需，用于调用 Qwen3.5）
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Zilliz Serverless 连接信息（必需）
MILVUS_HOST=https://your-cluster-id.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn
MILVUS_PORT=80
MILVUS_TOKEN=your-zilliz-token-here

# 向量集合名（如已有集合可改名避免冲突）
COLLECTION_NAME=pdf_rag_search

# 索引类型
INDEX=IVF_FLAT
```

### 5. 启动

```bash
python app.py
```

浏览器访问 `http://127.0.0.1:7860`。

## 使用方法

### 第一步：上传并编码 PDF

1. 在 **Document Management** 区域选择 PDF 文件
2. 在下拉框中选择文档（或保持 "All Documents"）
3. 点击 **Encode & Index** 按钮
4. 等待状态栏显示完成（17 页 PDF 约 10-20 秒，取决于网络和 Cohere API 响应速度）

### 第二步：提问

1. 在 **Ask Questions** 文本框中输入问题
2. 在下拉框中选择搜索范围（特定文档或 "All Documents" 跨文档搜索）
3. 点击 **Search & Answer** 按钮
4. 查看下方 **Retrieved Pages**（检索到的页面图片及分数）和 **Answer**（AI 回答）

### 管理数据

- 点击 **Clear All** 清空向量库和本地缓存
- 支持上传多个 PDF，搜索时可选择特定文档或跨文档搜索

## 配置参数

所有参数在 `config.py` 中定义，可通过 `.env` 覆盖：

| 参数 | .env 变量 | 默认值 | 说明 |
|---|---|---|---|
| `cohere_api_key` | `COHERE_API_KEY` | — | Cohere API 密钥（必需） |
| `embed_model` | — | `embed-v4.0` | Cohere Embedding 模型 |
| `embed_dim` | — | `1024` | 向量维度（可选 256 / 512 / 1024 / 1536） |
| `cohere_batch_size` | — | `96` | 每批编码页数（Cohere 单次最多 96 张图） |
| `top_k` | — | `3` | 检索返回的页面数 |
| `openrouter_api_key` | `OPENROUTER_API_KEY` | — | OpenRouter API 密钥（必需） |
| `generation_model` | — | `qwen/qwen3.5-397b-a17b` | LLM 模型（可换其他 OpenRouter 支持的视觉模型） |
| `llm_max_tokens` | — | `1024` | LLM 最大输出 token 数 |
| `llm_temperature` | — | `0.7` | LLM 生成温度 |
| `pdf_dpi` | — | `150` | PDF 渲染 DPI（值越高图片越清晰，但 Cohere 有 5MB 限制） |
| `max_image_size` | — | `1200` | 图片最大边长 px（受 Cohere 单张 5MB 限制） |
| `milvus_uri` | `MILVUS_HOST` | — | Zilliz 连接地址（必需） |
| `milvus_token` | `MILVUS_TOKEN` | — | Zilliz 认证 Token（必需） |
| `collection_name` | `COLLECTION_NAME` | `pdf_rag_search` | 向量集合名 |
| `index_type` | `INDEX` | `IVF_FLAT` | 向量索引类型 |

## 常见问题

### Q: 启动报错 `No module named 'cohere'` / `No module named 'fitz'`

```bash
pip install -r requirements.txt
```

确保使用 Python 3.10+ 并在正确的虚拟环境中安装。

### Q: 编码成功但搜索无结果 / 显示 0 vectors

Zilliz Serverless 存在少量统计延迟，不影响实际搜索。如搜索确实无结果，检查集合名是否与 `.env` 中的 `COLLECTION_NAME` 一致。

### Q: Cohere API 报错 401 / 429

- 401：API Key 无效，检查 `COHERE_API_KEY` 是否正确
- 429：超出免费额度，前往 [Cohere 控制台](https://dashboard.cohere.com) 查看用量

### Q: Zilliz 连接失败

检查 `.env` 中的 `MILVUS_HOST` 和 `MILVUS_TOKEN`。确认 Zilliz Serverless 集群已创建且处于运行状态。

### Q: Zilliz 报错 `Insert missed a field`

集合 schema 与代码不匹配（可能是旧集合）。在 `.env` 中更换 `COLLECTION_NAME` 为一个新名称即可。

### Q: OpenRouter 返回错误

检查 `OPENROUTER_API_KEY` 是否有效，账户是否有余额。Qwen3.5-397B-A17B 为大模型，可能排队，可重试。

### Q: 图片太大导致 Cohere 报错

Cohere 限制单张图片最大 5MB。系统默认将图片缩放到 1200px，如仍超出可调小 `config.py` 中的 `max_image_size`。

### Q: 检索结果不准确

- 尝试增大 `top_k`（如改为 5）
- 确保问题语言与 PDF 内容语言匹配
- 对于扫描件，适当提高 `pdf_dpi`（如 200 或 300）

## 开发指南

### 模块依赖关系

```
app.py (Flask Web 服务)
  ├── config.py (配置)
  ├── utils/pdf_processor.py (PDF → 图片, PyMuPDF)
  ├── core/vector_store.py (Zilliz 操作)
  ├── core/embedder.py (Cohere Embedding API)
  │     ├── utils/image_utils.py (图片转 base64 data URI)
  │     └── config.py
  ├── core/retriever.py (单向量检索)
  │     ├── core/embedder.py
  │     └── core/vector_store.py
  └── core/generator.py (LLM 生成)
        ├── utils/image_utils.py
        └── config.py
```

### 数据流

**文档入库**：PDF 文件 → PyMuPDF 渲染为图片 → base64 data URI → Cohere API 编码 → 1024 维向量 → 写入 Zilliz

**查询检索**：用户问题 → Cohere API 编码为向量 → Zilliz 内积搜索 → Top-K 页面图片 → base64 + 问题 → OpenRouter Qwen3.5 → AI 回答

### 扩展方向

- **更换 LLM 后端**：修改 `core/generator.py`，支持 OpenAI / Azure / 本地模型
- **更换 Embedding 模型**：修改 `core/embedder.py`，支持 [Voyage AI](https://www.voyageai.com) / [Jina AI](https://jina.ai) / Google Gemini Embedding
- **本地 Embedding 回退**：如需离线使用，可将 `CohereEmbedder` 替换回 ColQwen2 本地模型（需安装 PyTorch）
- **添加对话历史**：在 `app.py` 中增加会话管理，维护多轮对话上下文
- **前端升级**：`static/index.html` 为纯原生 HTML/JS，可按需引入 Vue/React 或替换为其他前端框架

## License

MIT
