# Multimodal RAG - PDF Intelligent Q&A

基于多模态 Embedding + Zilliz + Qwen 视觉理解的多模态 RAG 系统。支持 **Cohere / DashScope Embedding** 和 **DashScope / OpenRouter LLM** 双引擎切换。上传 PDF，用自然语言提问，系统自动检索最相关的页面并由 AI 生成回答。

与传统 RAG 不同，本系统**不做文本提取和 OCR**，而是直接将 PDF 页面当作图片处理，通过视觉 Embedding 模型编码，完整保留表格、图表、排版、手写批注等所有视觉信息。

## 工作原理

```
PDF 文档
  │
  ▼ (PyMuPDF, 150 DPI)
页面图片（不做任何文本提取）
  │
  ▼ (Embedding API, 云端调用)
  │   ├─ Cohere embed-v4.0 → 1 个 1024 维向量
  │   └─ DashScope tongyi-embedding-vision-plus → 1 个 1152 维向量
每页图片 → 1 个向量
  │
  ▼ (写入 Zilliz Serverless 云向量数据库)
doc_name + page_idx + vector
  │
  ══════════════ 用户提问时 ══════════════
  │
  ▼ (Embedding API 编码查询文本)
查询 → 1 个向量
  │
  ▼ (Zilliz 内积相似度搜索)
Top-K 最相关页面
  │
  ▼ (页面原始图片 + 问题 → 视觉大模型生成回答)
  │   ├─ DashScope: qwen3.5-flash / qwen3.5-plus / qwen3-vl-plus（国内推荐）
  │   └─ OpenRouter: qwen/qwen3.5-397b-a17b（海外可用）
AI 直接"看图"回答问题
```

**核心优势**：
- 无需本地 GPU，无需安装 PyTorch 或 poppler
- 所有 AI 计算通过云端 API 完成（Embedding + LLM 均支持 DashScope 和 OpenRouter）
- Embedding 和 LLM 引擎均可独立切换，国内用户推荐全部使用 DashScope（无需代理）
- 安装依赖只需几秒钟，9 个轻量 Python 包
- 面对扫描件 PDF、图文混排文档、含公式与表格的专业资料，都能完整保留所有视觉信息

## 技术栈

| 组件 | 技术 | 说明 |
|---|---|---|
| PDF → 图片 | PyMuPDF (fitz) | 纯 Python，无需 poppler，跨平台直接可用 |
| 图片/文本编码 | [Cohere embed-v4.0](https://docs.cohere.com/docs/embeddings) / [DashScope](https://dashscope.aliyun.com) | 云端多模态 Embedding，通过 `EMBED_PROVIDER` 切换引擎 |
| 向量数据库 | [Zilliz Serverless](https://cloud.zilliz.com) (云 Milvus) | IVF_FLAT 索引，内积相似度，零运维 |
| 生成模型 | [DashScope Qwen](https://dashscope.console.aliyun.com) / [OpenRouter](https://openrouter.ai) | 多模态视觉大模型，通过 `LLM_PROVIDER` 切换 |
| Web 界面 | Flask + 原生 HTML/JS | 轻量无框架依赖，无 telemetry，本地运行 |


### Embed 4：更懂企业数据的“大力士”

这Embed 4可不是简单的升级，它在前代Embed 3的基础上，狠狠地提升了一把。特别是在处理那些乱七八糟的非结构化数据时，简直就是一把好手。更厉害的是，它拥有高达128,000个token的超长上下文窗口，简单来说，就是能记住更多东西，理论上能给大概200页的文档生成嵌入！

Cohere自己也说了，之前的嵌入模型在理解企业那些复杂、多格式的数据时，总是差口气，导致企业得花大量时间做数据预处理，效果还不咋地。Embed 4就是为了解决这个问题而生的，帮助企业员工从一大堆乱七八糟的信息里，快速找到关键信息。

## 项目结构

```
D:/PDF-AI/
├── .env                    # 密钥和连接配置（不入库）
├── .env.example            # 配置模板
├── .gitignore
├── requirements.txt        # Python 依赖（9 个包，无 PyTorch）
├── config.py               # 配置中心，从 .env 加载
├── app.py                  # Flask Web 服务 + API 路由
├── static/
│   └── index.html          # 前端页面
│
├── core/
│   ├── embedder.py         # 双引擎 Embedding（Cohere / DashScope）+ 工厂函数
│   ├── vector_store.py     # Zilliz 向量库：集合管理、插入、搜索
│   ├── retriever.py        # 单向量检索：编码查询 → 搜索 Zilliz → Top-K 页面
│   └── generator.py        # 双引擎 LLM 生成（DashScope / OpenRouter）
│
└── utils/
    ├── pdf_processor.py    # PDF 转图片（PyMuPDF 封装，无需 poppler）
    └── image_utils.py      # 图片转 base64（data URI 供 Cohere，纯 base64 供 DashScope）
```

## API 接口

| 方法 | 路径 | 说明 | 请求体 |
|---|---|---|---|
| GET | `/api/docs` | 获取所有已上传的文档列表 | 无 |
| POST | `/api/upload` | 上传 PDF 文件 | `multipart/form-data`，字段 `file` |
| POST | `/api/encode` | 编码已上传的 PDF 并写入向量库 | `{"doc_name": "xxx.pdf"}` |
| POST | `/api/search` | 检索相关页面并生成回答 | `{"question": "...", "doc_name": "xxx.pdf"}` |
| POST | `/api/clear` | 清空所有数据（Zilliz + 本地文件） | 无 |

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

## 功能流程详解

### 数据存储架构

系统有三级存储，各司其职：

```
┌─────────────────────────────────────────────────────┐
│  data/uploads/  (本地磁盘)                            │
│  ├── report.pdf          原始 PDF 文件，持久保存        │
│  ├── paper.pdf           服务重启/页面刷新后仍在        │
│  └── ...                                             │
├─────────────────────────────────────────────────────┤
│  Zilliz Cloud  (云端向量库)                           │
│  集合: pdf_rag_cohere / pdf_rag_dashscope（按引擎自动选择）│
│  每条记录: doc_name + page_idx + vector(由引擎决定维度)│
│  已 encode 的文档向量永久保存，不受重启影响             │
├─────────────────────────────────────────────────────┤
│  内存 (_image_cache)  (进程级缓存)                     │
│  doc_name → [PIL.Image, PIL.Image, ...]              │
│  加速图片访问，按需从 PDF 重新加载                      │
└─────────────────────────────────────────────────────┘
```

**为什么刷新页面后仍可搜索**：
- PDF 文件保存在 `data/uploads/`，不是临时文件
- 向量数据存在 Zilliz 云端，不在本地内存
- 页面加载时，前端调用 `GET /api/docs` 扫描 `data/uploads/` 获取文档列表
- 搜索时，如果图片不在内存缓存，自动从 PDF 文件按需加载

### Zilliz 集合 Schema

```
集合名: pdf_rag_cohere (Cohere) / pdf_rag_dashscope (DashScope)
  (可通过 .env 的 COHERE_COLLECTION_NAME / DASHSCOPE_COLLECTION_NAME 配置)

字段:
  id       INT64       自增主键
  doc_name VARCHAR(256) PDF 文件名，用于区分不同文档
  page_idx INT64       页码（从 0 开始）
  vector   FLOAT_VECTOR(N)     N 由引擎决定：Cohere 1024 / DashScope 1152

索引:
  字段: vector
  类型: IVF_FLAT
  相似度: IP (内积)
  参数: nlist=128, nprobe=10
```

### 流程一：文档入库（Upload → Encode → Index）

```
用户选择 PDF 文件
  │
  ▼ POST /api/upload
Flask 接收文件，保存到 data/uploads/{filename}.pdf
  ├── 同名文件会被覆盖
  └── 清除该文档的内存图片缓存
  │
  ▼ 前端下拉框显示文档名
  │
用户点击 "Encode & Index"
  │
  ▼ POST /api/encode {doc_name: "xxx.pdf"}
  │
  ├── Step 1: PDF → 图片
  │   PyMuPDF (fitz) 打开 PDF，逐页渲染为 RGB 图片
  │   DPI 由 config.pdf_dpi 控制（默认 150）
  │   图片缓存在内存 _image_cache[doc_name]
  │
  ├── Step 2: 图片 → 向量 (Embedding API)
  │   将每张图片缩放到 max_image_size (默认 1200px)
  │   ├─ DashScope: 转为 JPEG base64，按 8 张/批调用 MultiModalEmbedding.call()
  │   │   模型: tongyi-embedding-vision-plus，产出 1152 维向量
  │   └─ Cohere: 转为 JPEG base64 data URI，按 96 张/批调用 embed()
  │       模型: embed-v4.0，input_type=search_document，产出 1024 维向量
  │   每页产出 1 个向量（维度由引擎决定）
  │
  └── Step 3: 写入 Zilliz
      每页一条记录: {doc_name, page_idx, vector}
      如果该文档之前已 encode 过，向量会重复插入
      （建议 clear 后重新 encode，或后续优化为 upsert）
```

### 流程二：提问检索（Search → Retrieve → Generate）

```
用户输入问题，点击 "Search & Answer"
  │
  ▼ POST /api/search {question: "...", doc_name: "xxx.pdf" | "__all__"}
  │
  ├── Step 1: 查询编码 (Embedding API)
  │   ├─ DashScope: MultiModalEmbedding.call(input=[{text: ...}]) → 1152 维
  │   └─ Cohere: embed(texts=[...], input_type="search_query") → 1024 维
  │   问题文本 → 1 个向量
  │
  ├── Step 2: 向量搜索 (Zilliz)
  │   用查询向量在 Zilliz 中做内积相似度搜索
  │   如果指定了 doc_name，添加过滤条件 filter='doc_name == "xxx"'
  │   如果选 "All Documents"，不加过滤，跨文档搜索
  │   返回 Top-K (默认 3) 条结果，每条包含 doc_name + page_idx + score
  │
  ├── Step 3: 加载页面图片
  │   根据 (doc_name, page_idx) 获取页面图片:
  │   优先从内存缓存 _image_cache 读取
  │   缓存未命中时，从 data/uploads/{doc_name} 用 PyMuPDF 按需加载
  │   图片转 PNG base64，返回给前端展示
  │
  └── Step 4: LLM 生成回答
      构建多模态消息:
      content = [
        {type: "image_url", image_url: {url: "data:image/png;base64,..."}},
        ...每张检索到的页面图片...
        {type: "text", text: "Above are N retrieved document pages. Question: ..."}
      ]
      ├─ DashScope (默认): 调用 dashscope.aliyuncs.com OpenAI 兼容接口
      │   模型: qwen3.5-flash / qwen3.5-plus / qwen3-vl-plus 等
      └─ OpenRouter: 调用 openrouter.ai
          模型: qwen/qwen3.5-397b-a17b
      超时: 120 秒
      LLM 直接"看"页面图片生成回答
```

### 流程三：清空数据（Clear）

```
用户点击 "Clear All"
  │
  ▼ POST /api/clear
  │
  ├── 删除 Zilliz 集合并重建空集合
  ├── 删除 data/uploads/ 下所有 PDF 文件
  └── 清空内存图片缓存
```

## 快速开始

### 1. 环境要求

- Python 3.10+
- 无需 GPU、无需 poppler、无需 CUDA
- 网络能访问 DashScope API、Zilliz Cloud（国内用户推荐全选 DashScope，只需一个 API Key）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

共 9 个轻量包（cohere、dashscope、pymilvus、openai、PyMuPDF、pillow、flask、numpy、python-dotenv），安装通常在 1 分钟内完成。

### 3. 获取 API Key

**DashScope**（Embedding + LLM，默认引擎，国内推荐）：前往 [https://dashscope.console.aliyun.com](https://dashscope.console.aliyun.com) 获取 API Key。一个 Key 同时用于 Embedding 和视觉理解。

**Cohere**（Embedding，可选引擎）：前往 [https://dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys) 注册获取，有免费额度。

**OpenRouter**（LLM，可选引擎）：前往 [https://openrouter.ai/settings/keys](https://openrouter.ai/settings/keys) 注册获取。

**Zilliz Cloud**（向量数据库，必需）：前往 [https://cloud.zilliz.com](https://cloud.zilliz.com) 创建 Serverless 集群，获取连接地址和 Token。

### 4. 配置 .env

复制 `.env.example` 为 `.env`，填入实际值：

```env
# Embedding 引擎（dashscope 或 cohere）
EMBED_PROVIDER=dashscope

# DashScope API Key（Embedding + LLM 共用，国内推荐）
DASHSCOPE_API_KEY=your-dashscope-api-key

# Cohere API Key（使用 Cohere Embedding 时必需）
COHERE_API_KEY=your-cohere-api-key

# LLM 引擎（dashscope 或 openrouter）
LLM_PROVIDER=dashscope

# DashScope 视觉模型（qwen3.5-flash / qwen3.5-plus / qwen3-vl-plus）
DASHSCOPE_VL_MODEL=qwen3.5-flash

# OpenRouter API Key（使用 OpenRouter LLM 时必需）
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Zilliz Serverless 连接信息（必需）
MILVUS_HOST=https://your-cluster-id.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn
MILVUS_TOKEN=your-zilliz-token-here

# 向量集合名（每个引擎独立，不会冲突）
COHERE_COLLECTION_NAME=pdf_rag_cohere
DASHSCOPE_COLLECTION_NAME=pdf_rag_dashscope

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

1. 在 **文档管理** 区域选择 PDF 文件
2. 点击 **索引入库** 按钮（自动编码刚上传的文件）
3. 等待状态栏显示完成（编码速度取决于 API 引擎和网络）

### 第二步：提问

1. 在 **提问** 区域输入问题
2. 通过下拉框选择搜索范围（特定文档或 **全部文档** 跨文档搜索）
3. 点击 **搜索回答** 按钮
4. 查看下方 **检索到的页面**（页面图片及相似度分数）和 **回答**（AI 生成）

## 配置参数

所有参数在 `config.py` 中定义，可通过 `.env` 覆盖：

| 参数 | .env 变量 | 默认值 | 说明 |
|---|---|---|---|
| `embed_provider` | `EMBED_PROVIDER` | `dashscope` | Embedding 引擎：`dashscope`（国内推荐）或 `cohere` |
| `dashscope_api_key` | `DASHSCOPE_API_KEY` | — | DashScope API 密钥（Embedding + LLM 共用） |
| `cohere_api_key` | `COHERE_API_KEY` | — | Cohere API 密钥（使用 Cohere 时必需） |
| `embed_model` | — | 由 provider 决定 | DashScope: `tongyi-embedding-vision-plus` / Cohere: `embed-v4.0` |
| `embed_dim` | — | 由 provider 决定 | DashScope: 1152 / Cohere: 1024 |
| `cohere_batch_size` | — | `96` | Cohere 每批编码页数（DashScope 固定 8 张/次） |
| `llm_provider` | `LLM_PROVIDER` | `dashscope` | LLM 引擎：`dashscope`（国内推荐）或 `openrouter` |
| `dashscope_vl_model` | `DASHSCOPE_VL_MODEL` | `qwen3.5-flash` | DashScope 视觉模型（`qwen3.5-flash` / `qwen3.5-plus` / `qwen3-vl-plus`） |
| `openrouter_api_key` | `OPENROUTER_API_KEY` | — | OpenRouter API 密钥（使用 OpenRouter 时必需） |
| `generation_model` | — | 由 provider 决定 | DashScope: `qwen3.5-flash` / OpenRouter: `qwen/qwen3.5-397b-a17b` |
| `top_k` | — | `3` | 检索返回的页面数 |
| `llm_max_tokens` | — | `1024` | LLM 最大输出 token 数 |
| `llm_temperature` | — | `0.7` | LLM 生成温度 |
| `pdf_dpi` | — | `150` | PDF 渲染 DPI |
| `max_image_size` | — | `1200` | 图片最大边长 px |
| `milvus_uri` | `MILVUS_HOST` | — | Zilliz 连接地址（必需） |
| `milvus_token` | `MILVUS_TOKEN` | — | Zilliz 认证 Token（必需） |
| `cohere_collection_name` | `COHERE_COLLECTION_NAME` | `pdf_rag_cohere` | Cohere 引擎的向量集合名 |
| `dashscope_collection_name` | `DASHSCOPE_COLLECTION_NAME` | `pdf_rag_dashscope` | DashScope 引擎的向量集合名 |
| `index_type` | `INDEX` | `IVF_FLAT` | 向量索引类型 |

## 常见问题

### Q: 启动报错 `No module named 'cohere'` / `No module named 'fitz'`

```bash
pip install -r requirements.txt
```

确保使用 Python 3.10+ 并在正确的虚拟环境中安装。

### Q: 编码成功但搜索无结果 / 显示 0 vectors

检查集合名是否正确，切换 Embedding 引擎后需重新编码文档。

### Q: Cohere API 报错 401 / 429

- 401：API Key 无效，检查 `COHERE_API_KEY` 是否正确
- 429：超出免费额度，前往 [Cohere 控制台](https://dashboard.cohere.com) 查看用量
- 403：网络被拦截（国内常见），切换为 DashScope 引擎：`EMBED_PROVIDER=dashscope`

### Q: DashScope API 报错

- 检查 `DASHSCOPE_API_KEY` 是否正确
- 前往 [DashScope 控制台](https://dashscope.console.aliyun.com) 查看用量和余额

### Q: 切换 Embedding 引擎后搜索报错维度不匹配

不会发生。每个引擎使用独立的集合（`pdf_rag_cohere` / `pdf_rag_dashscope`），切换引擎后自动连接对应集合，数据互不干扰。但每个集合的文档需要独立 encode。

### Q: Zilliz 连接失败

检查 `.env` 中的 `MILVUS_HOST` 和 `MILVUS_TOKEN`。确认 Zilliz Serverless 集群已创建且处于运行状态。

### Q: Zilliz 报错 `Insert missed a field`

集合 schema 与代码不匹配（可能是旧集合）。在 `.env` 中更换 `COHERE_COLLECTION_NAME` 或 `DASHSCOPE_COLLECTION_NAME` 为一个新名称即可。

### Q: OpenRouter 返回错误

检查 `OPENROUTER_API_KEY` 是否有效，账户是否有余额。如国内访问不稳定，可切换 `LLM_PROVIDER=dashscope`。

### Q: DashScope 视觉模型返回错误

- 检查 `DASHSCOPE_API_KEY` 是否正确（与 Embedding 共用同一个 Key）
- 确认 `DASHSCOPE_VL_MODEL` 模型名称有效（`qwen3.5-flash` / `qwen3.5-plus` / `qwen3-vl-plus`）

### Q: 图片太大导致 Cohere 报错

Cohere 限制单张图片最大 5MB。系统默认将图片缩放到 1200px，如仍超出可调小 `config.py` 中的 `max_image_size`。

### Q: 检索结果不准确

- 尝试增大 `top_k`（如改为 5）
- 确保问题语言与 PDF 内容语言匹配
- 对于扫描件，适当提高 `pdf_dpi`（如 200 或 300）

## 开发指南

### 模块依赖关系

```
app.py (Flask Web 服务, 5 个 API 路由)
  ├── config.py (配置中心, 从 .env 加载 8 个变量)
  ├── utils/pdf_processor.py (PDF → 图片, PyMuPDF)
  ├── core/vector_store.py (Zilliz 操作)
  ├── core/embedder.py (双引擎 Embedding: Cohere / DashScope, create_embedder() 工厂)
  │     ├── utils/image_utils.py (图片 → base64, data URI / 纯 base64 两种格式)
  │     └── config.py
  ├── core/retriever.py (单向量检索)
  │     ├── core/embedder.py
  │     └── core/vector_store.py
  └── core/generator.py (双引擎 LLM: DashScope / OpenRouter)
        ├── utils/image_utils.py
        └── config.py
```

### 各模块职责

**`config.py`** — 配置中心。`Settings` dataclass 定义所有参数，`_load_env()` 从 `.env` 读取变量覆盖默认值，`_resolve_provider()` 根据 `EMBED_PROVIDER` 和 `LLM_PROVIDER` 分别解析 Embedding 和 LLM 的活跃配置。全局单例 `settings` 供所有模块导入。

**`utils/pdf_processor.py`** — `pdf_to_images(pdf_path, dpi)` 用 PyMuPDF 将 PDF 逐页渲染为 PIL Image 列表。`PdfProcessingError` 自定义异常用于友好错误提示。无需 poppler。

**`utils/image_utils.py`** — `image_to_data_uri(img, max_size, fmt)` 将 PIL Image 缩放后转为 base64 data URI（Cohere 使用）。`pil_to_base64(img, max_size, fmt)` 返回纯 base64 字符串（DashScope 使用）。

**`core/embedder.py`** — 双引擎 Embedding 封装。`BaseEmbedder` 定义接口，`CohereEmbedder` 封装 Cohere embed-v4.0 API，`DashScopeEmbedder` 封装 DashScope tongyi-embedding-vision-plus API。`create_embedder()` 工厂函数根据 `settings.embed_provider` 返回对应实例。两个引擎均实现 `encode_images()` 和 `encode_query()`，并内置速率限制。

**`core/vector_store.py`** — `VectorStore` 封装 Zilliz 操作。首次连接自动创建集合。`insert_pages()` 按页插入向量，`search()` 支持按 doc_name 过滤。用 MilvusClient 的 search 方法做内积搜索。

**`core/retriever.py`** — `Retriever` 组合 embedder + vector_store。`retrieve()` 依次调用编码查询 → 搜索 Zilliz → 返回 `RetrievalResult` 列表。单向量检索，无 MaxSim 聚合。

**`core/generator.py`** — 双引擎 LLM 生成。`AnswerGenerator` 根据 `settings.llm_provider` 初始化 OpenAI 兼容客户端：DashScope 走 `dashscope.aliyuncs.com/compatible-mode/v1`，OpenRouter 走 `openrouter.ai/api/v1`。两者消息格式完全相同，`generate()` 构建多模态消息（图片 base64 + 文本 prompt），发给视觉大模型生成回答。

**`app.py`** — Flask 服务。`data/uploads/` 持久化 PDF 文件，`_image_cache` 内存缓存页面图片（按需从 PDF 加载）。5 个 API 路由，每个都有 try/except + logging。组件懒加载（首次使用时初始化）。

### 扩展方向

- **添加新 LLM 后端**：在 `core/generator.py` 中添加新的 `base_url` 分支，支持 OpenAI / Azure / 本地模型
- **添加新 Embedding 引擎**：在 `core/embedder.py` 中继承 `BaseEmbedder`，实现 `encode_images()` 和 `encode_query()`，并在 `create_embedder()` 工厂中注册
- **本地 Embedding 回退**：如需离线使用，可继承 `BaseEmbedder` 接入 ColQwen2 本地模型（需安装 PyTorch）
- **添加对话历史**：在 `app.py` 中增加会话管理，维护多轮对话上下文
- **前端升级**：`static/index.html` 为纯原生 HTML/JS，可按需引入 Vue/React 或替换为其他前端框架

## License

MIT
