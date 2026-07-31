# VidWise

视频智能平台 — 从视频中提炼可检索、可对话的知识。

## 能力

- **视频/音频下载**：通过 yt-dlp 下载视频或提取音频
- **语音转文字 (ASR)**：Whisper / Faster-Whisper 模型，支持 VAD 降噪和长音频分片
- **智能文本格式化**：LLM 驱动的错别字修正、繁简转换、段落划分
- **RAG 知识库**：文本自动分段 → 向量化 → 存入 Qdrant，支持检索和重排序
- **多轮对话问答**：基于视频知识库和对话历史的智能问答，会话持久化到 MySQL
- **用户记忆**：从多轮对话中沉淀跨会话用户事实，可在问答时作为长期上下文
- **运行时能力清单**：网关输出前端 manifest，Web UI 根据 ASR/RAG/LLM/Memory 等能力自动降级或禁用功能
- **MCP Server**：通过 MCP 协议对外暴露工具，可被 Claude Desktop 等客户端发现和调用

## 架构

```
浏览器 / API 客户端
        │
        ▼
┌─────────────── Gin API Gateway (:8080) ───────────────┐
│  /extract  /format  /chat/query  /agent/turn          │
│  /video/process  /tasks  /api/capabilities            │
│  /rag/health  /user/facts  /mcp (SSE :8082)           │
│                                                        │
│  ┌─ Eino Agent/Tool 编排 ─┐  ┌─ Chat Session 管理 ─┐  │
│  │ download → asr → format│  │ MySQL (GORM)         │  │
│  │ → rag_index → rag_query│  │ 会话/消息持久化       │  │
│  └────────────────────────┘  └──────────────────────┘  │
└────────────────────────────────────────────────────────┘
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ ASR Service  │ │ Embedding    │ │ Video Summary│
│ (:8001)      │ │ Service      │ │ (:8002)     │
│ Whisper/     │ │ (:8003)      │ │ Marlin-2B    │
│ FasterWhisper│ │ Embed API +  │ │              │
│ + Silero VAD │ │ Rerank API   │ │              │
└──────────────┘ └──────┬───────┘ └──────────────┘
                        │
                 ┌──────▼───────┐
                 │    Qdrant    │
                 │ 向量数据库    │
                 │  (:6334)     │
                 └──────────────┘
```

## 快速开始

### 前置条件

- Go 1.26+
- Python 3.10+
- ffmpeg, yt-dlp
- Ollama (本地 LLM) 或 OpenAI API Key
- Docker (可选，用于 MySQL + Qdrant)

### 1. 安装系统依赖

```bash
make deps
```

### 2. 安装 Python 依赖

```bash
python3 -m venv .venv && source .venv/bin/activate
make deps-python
make deps-embedding
```

### 3. 下载模型

```bash
# ASR 模型
git clone https://huggingface.co/openai/whisper-small ./models/whisper-small

# Embedding 模型 (二选一)
huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir ./models/qwen3-embedding
huggingface-cli download BAAI/bge-m3 --local-dir ./models/bge-m3

# Rerank 模型 (本地重排时需要)
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir ./models/bge-reranker-v2-m3
```

如果使用阿里云模型 API，可以跳过对应的本地模型下载，在
`config.yaml` 中把 `asr.model.provider` 或 `embedding.provider` 改为
`aliyun`，并设置 `DASHSCOPE_API_KEY`。
如果使用硅基流动 Embedding/Rerank API，可以跳过 embedding/rerank 本地模型下载，把
`embedding.provider` 或 `rerank.provider` 改为 `siliconflow`，并设置 `SILICONFLOW_API_KEY`；
默认接口地址为 `https://api.siliconflow.cn/v1`，默认 `qwen` 快捷模型会映射到
`Qwen/Qwen3-Embedding-0.6B`。`embedding.dimensions` 只会发送给 SiliconFlow
的 Qwen3 embedding 模型；使用 `BAAI/bge-m3` 等模型时适配器会自动省略该参数。
专用重排模型可使用 `BAAI/bge-reranker-v2-m3`。
如果使用阿里云专用重排模型，把 `rerank.provider` 改为 `aliyun`；`qwen3-rerank`
走 OpenAI-compatible `/reranks` 接口，`gte-rerank-v2` 走 DashScope
`/services/rerank/text-rerank/text-rerank` 接口。
如果使用科大讯飞录音文件转写大模型，把 `asr.model.provider` 改为
`xfyun`，并设置 `XFYUN_APP_ID`、`XFYUN_API_KEY` 和 `XFYUN_API_SECRET`。
如果使用百度智能云短语音识别，把 `asr.model.provider` 改为 `baidu`，
并设置 `BAIDU_ASR_API_KEY` 和 `BAIDU_ASR_SECRET_KEY`；服务会用 ffmpeg
把本地音频转为 16 kHz 单声道 WAV，并按百度短语音接口的 60 秒限制分片提交。

### 4. 配置

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml 填入你的 LLM API Key 或 Ollama 地址
```

### 5. 启动基础设施

```bash
# 启动 MySQL + Qdrant
docker compose -f docker-compose.example.yaml up -d

# 或单独安装
brew install mysql qdrant
```

### 6. 启动服务

一键启动所有服务：

```bash
make run-all-bg
```

或分别启动：

```bash
make run-embedding  # Embedding/Rerank 服务 :8003
make run-asr        # ASR 语音转文字服务 :8001
make run            # API 网关 :8080

# 打开浏览器访问
open http://localhost:8080
```

Web UI 默认打开可对话的 Agent 问答界面，侧边栏会从 `/api/capabilities`
同步当前后端能力，并按可用性展示或禁用功能。页面包含同步提取、视频任务、
知识库 source 管理、上传索引、文本格式化、任务列表和用户记忆查看。

停止所有服务：

```bash
make stop-all
```

## API

### 视频提取 (同步)

```bash
curl -X POST http://localhost:8080/extract \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/video","name":"demo","type":"text"}' \
  -o demo.txt
```

`url` 可以是纯视频链接，也可以是 bilibili、抖音、小红书复制出来的完整分享文案；当分享文案里能解析出标题时，`name` 可省略并自动用标题生成安全文件名。

`type` 支持的值：`video` | `audio` | `text` | `transcript` | `summary` | `video_summary`

文本提取完成后会在 RAG 可用且请求带有 `user_id` 或 `session_id` 时自动索引到知识库。
同样的能力也暴露在 Web UI 的「同步提取」页；页面会按 `/api/capabilities`
返回的 `extract_types` 禁用不可用类型。

### 运行时能力清单

```bash
curl http://localhost:8080/api/capabilities
```

返回内容包括：

- `capabilities`：ASR、RAG、LLM、Memory、Embedding、Rerank 等运行时能力状态
- `features`：Web UI 功能、对应 tab、依赖能力和后端路由
- `extract_types`：同步提取支持的类型及其依赖能力
- `video_process_steps`：异步视频处理 DAG 步骤名称
- `tools`：当前注册到工具中心的工具名

### 会话式问答

```bash
# 创建新会话并提问（自动创建会话）
curl -X POST http://localhost:8080/chat/query \
  -H "Content-Type: application/json" \
  -d '{"query":"这个视频讲了什么？"}'

# 在已有会话中追问
curl -X POST http://localhost:8080/chat/query \
  -H "Content-Type: application/json" \
  -d '{"session_id":"<session_id>","query":"具体怎么操作？"}'

# 获取会话列表
curl http://localhost:8080/chat/sessions

# 获取某个会话的完整对话
curl http://localhost:8080/chat/session/<session_id>
```

### 用户记忆

```bash
curl "http://localhost:8080/user/facts?user_id=<user_id>"
```

当 MySQL memory store 可用时，该接口返回用户长期事实列表；Web UI 的「用户记忆」页也使用同一接口。

### 健康检查

```bash
curl http://localhost:8080/health             # 网关与 canonical capabilities
curl http://localhost:8080/ready              # 生产流量就绪状态
curl http://localhost:8080/api/capabilities   # 前端能力 manifest
curl http://localhost:8001/health             # ASR
curl http://localhost:8003/health             # Embedding
curl http://localhost:8080/rag/health         # RAG 兼容健康状态
```

## 配置说明

完整配置见 `config.example.yaml`：

| 配置段 | 说明 |
|--------|------|
| `server` | 网关监听地址 |
| `asr` | ASR 服务地址 + 模型配置 (whisper/faster-whisper/aliyun/xfyun/baidu) |
| `llm` | LLM 提供商 (openai/ollama/deepseek) + 格式化参数 |
| `mysql` | MySQL 连接串 (用于会话、记忆、source registry 和异步任务持久化，可选) |
| `qdrant` | Qdrant 向量数据库地址 |
| `embedding` | Embedding 服务配置 (本地 qwen/bge 或 aliyun/siliconflow API) |
| `rerank` | 专用重排序配置 (本地 CrossEncoder 或 aliyun/siliconflow Rerank API) |
| `mcp` | MCP Server 开关和端口 |
| `task` | 异步任务保留策略；未配置 MySQL 时使用本地 JSON fallback |

## 项目结构

```
vidwise/
├── main.go                  # 入口: gateway / worker 模式
├── cmd/                     # CLI 子命令: download, audio, video
├── internal/
│   ├── agent/               # 视频处理 pipeline (graph.go)
│   ├── appconfig/           # 配置加载与默认值
│   ├── asr/                 # ASR HTTP 客户端
│   ├── chat/                # 会话/消息模型与持久化 (GORM)
│   ├── extractor/           # 旧版 /extract 端点逻辑
│   ├── mcp/                 # MCP Server (mcp-go)
│   ├── model/               # Embed/Rerank HTTP 客户端
│   ├── paragraph/           # LLM 文本格式化
│   ├── rag/                 # RAG: chunker, indexer, retriever
│   ├── server/              # Gin 路由、中间件、Handler、前端 manifest、Web UI
│   ├── storage/             # MySQL + Qdrant 客户端
│   ├── task/                # 异步任务模型
│   ├── tool/                # Eino 工具注册中心 + 各工具实现
│   └── user/                # 用户模型 (保留)
├── asr_service/             # Python ASR 服务 (FastAPI)
├── embedding_service/       # Python Embedding/Rerank 服务
├── video_summary_service/   # Python 视频理解服务
├── config.example.yaml
├── docker-compose.example.yaml
└── Makefile
```

## License

MIT
