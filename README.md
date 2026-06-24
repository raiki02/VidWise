# VidWise

视频智能平台 — 从视频中提炼可检索、可对话的知识。

## 能力

- **视频/音频下载**：通过 yt-dlp 下载视频或提取音频
- **语音转文字 (ASR)**：Whisper / Faster-Whisper 模型，支持 VAD 降噪和长音频分片
- **智能文本格式化**：LLM 驱动的错别字修正、繁简转换、段落划分
- **RAG 知识库**：文本自动分段 → 向量化 → 存入 Qdrant，支持检索和重排序
- **多轮对话问答**：基于视频知识库和对话历史的智能问答，会话持久化到 MySQL
- **MCP Server**：通过 MCP 协议对外暴露工具，可被 Claude Desktop 等客户端发现和调用

## 架构

```
浏览器 / API 客户端
        │
        ▼
┌─────────────── Gin API Gateway (:8080) ───────────────┐
│  /extract  /format  /chat/query  /chat/sessions       │
│  /video/process  /rag/health  /mcp (SSE :8082)       │
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
│ FasterWhisper│ │ Qwen/BGE     │ │              │
│ + Silero VAD │ │ embed+rerank │ │              │
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
```

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

`type` 支持的值：`video` | `audio` | `text` | `transcript` | `summary` | `video_summary`

文本提取完成后自动索引到 RAG 知识库。

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

### 健康检查

```bash
curl http://localhost:8080/health       # 网关
curl http://localhost:8001/health       # ASR
curl http://localhost:8003/health       # Embedding
curl http://localhost:8080/rag/health   # RAG 状态
```

## 配置说明

完整配置见 `config.example.yaml`：

| 配置段 | 说明 |
|--------|------|
| `server` | 网关监听地址 |
| `asr` | ASR 服务地址 + 模型配置 (whisper/faster-whisper) |
| `llm` | LLM 提供商 (openai/ollama/deepseek) + 格式化参数 |
| `mysql` | MySQL 连接串 (用于会话持久化，可选) |
| `qdrant` | Qdrant 向量数据库地址 |
| `embedding` | Embedding 服务配置 (qwen/bge 模型) |
| `rerank` | 重排序参数 |
| `mcp` | MCP Server 开关和端口 |

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
│   ├── server/              # Gin 路由、中间件、Handler、Web UI
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
