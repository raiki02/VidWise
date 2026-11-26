# VidWise (MVP)

视频智能平台 — 最小可运行 MVP：给一条视频 URL，自动完成「下载 → 转写 → 向量化入库」，
随后可基于知识库多轮问答。

核心：**CloudWeGo eino ReAct Agent**。ASR 与 Embedding 是独立的 Python 服务，
通过 HTTP 被封装成 eino 工具，由 Agent 按需调用。

## 能力

- **下载**：yt-dlp 下载音频（`download_audio` 工具）
- **语音转文字**：`transcribe_audio` 工具 → ASR 服务 (:8001)，支持 whisper /
  faster-whisper / 阿里云 / 讯飞 / 百度五种 provider
- **知识库入库**：`index_transcript` 工具 → 文本切分 → Embedding 服务 (:8003) 向量化 →
  Qdrant 存储
- **知识库检索**：`search_knowledge` 工具 → 向量相似度检索
- **多轮会话问答**：Agent 自动决定调用哪些工具；会话历史持久化到 MySQL（未配置时
  退化为内存存储）

已按 MVP 范围**移除**：Web 搜索、MCP、视频摘要、重排序（rerank）、段落格式化、
异步任务、能力清单接口。

## 架构

```
浏览器 / API 客户端
        │
        ▼
┌─────────────── Gin Gateway (:8080) ───────────────┐
│  GET  /health                                      │
│  POST /agent/chat                                  │
│  GET  /          (内嵌精简单页 UI)                  │
│                                                    │
│  ┌─ eino ReAct Agent ──────────────────────────┐  │
│  │  download_audio → transcribe_audio          │  │
│  │  → index_transcript → search_knowledge      │  │
│  └─────────────────────────────────────────────┘  │
│  会话历史: MySQL(GORM) / 内存回退                  │
└────────────────────────────────────────────────────┘
        │                  │
        ▼                  ▼
┌──────────────┐   ┌──────────────┐        ┌──────────┐
│ ASR Service  │   │ Embedding    │        │  Qdrant  │
│ (:8001)      │   │ Service      │        │ (:6334)  │
│ /transcribe  │   │ (:8003)      │        │ gRPC     │
│ 5 providers  │   │ /embed       │        │ Cosine   │
└──────────────┘   │ 3 providers  │        └──────────┘
                   └──────────────┘
```

LLM 走 eino-ext 的 OpenAI 兼容接口：`openai` / `ollama` / `deepseek`。

## 快速开始

### 前置条件

- Go 1.26+
- Python 3.10+（Python 服务）
- ffmpeg、yt-dlp（`make deps` 会检查/安装）
- LLM：OpenAI API Key，或本地 Ollama，或 DeepSeek API Key
- Qdrant（本地二进制或 Docker）
- MySQL（可选，用于会话持久化）

### 1. 系统与 Python 依赖

```bash
make deps
python3 -m venv .venv && source .venv/bin/activate
make deps-python      # ASR 服务依赖
make deps-embedding   # Embedding 服务依赖
```

### 2. 模型（本地 provider 时需要）

```bash
# ASR
git clone https://huggingface.co/openai/whisper-small ./models/whisper-small

# Embedding
huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir ./models/qwen3-embedding
```

云端 provider 可跳过下载：

| Provider | 配置 | 环境变量 |
|----------|------|----------|
| 阿里云 ASR | `asr.model.provider: aliyun`（可用 `dashscope`） | `DASHSCOPE_API_KEY` |
| 讯飞 ASR | `asr.model.provider: xfyun`（可用 `iflytek`） | `XFYUN_APP_ID` / `XFYUN_API_KEY` / `XFYUN_API_SECRET` |
| 百度 ASR | `asr.model.provider: baidu` | `BAIDU_ASR_API_KEY` / `BAIDU_ASR_SECRET_KEY` |
| 阿里云 Embedding | `embedding.provider: aliyun` | `DASHSCOPE_API_KEY` |
| 硅基流动 Embedding | `embedding.provider: siliconflow` | `SILICONFLOW_API_KEY` |

### 3. 配置与基础设施

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml：llm.provider/model/api_key，qdrant.vector_dim 需与 embedding 维度一致

docker compose -f docker-compose.example.yaml up -d   # MySQL + Qdrant（可选 MySQL）
# 或单独安装: brew install qdrant mysql
```

### 4. 启动

```bash
make run-all-bg        # 一键启动 (embedding :8003 → asr :8001 → gateway :8080)
# 或分别启动
make run-embedding
make run-asr
make run

open http://localhost:8080   # 精简单页 UI
```

停止：`make stop-all`

## API

### `POST /agent/chat`

```bash
# 首次提问（新会话；session_id 由服务端生成并返回）
curl -X POST http://localhost:8080/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"帮我分析 https://example.com/video.mp4"}'

# 追问（携带 session_id）
curl -X POST http://localhost:8080/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"<session_id>","message":"视频里提到了哪些步骤？"}'
```

请求：`{"session_id": "...(可选)", "message": "..."}`
响应：`{"session_id": "...", "reply": "..."}`

- `400`：JSON 非法或 `message` 为空
- `404`：`session_id` 不存在
- `500`：Agent 执行失败

### `GET /health`

网关健康检查。Python 服务各自也有 `/health`（:8001 / :8003）。

### `GET /`

内嵌精简单页 UI：消息列表 + 输入框，`session_id` 存于 localStorage，支持「新对话」。

## 端到端冒烟

1. 启动 Qdrant、ASR、Embedding、gateway（`make run-all-bg`）
2. 浏览器打开 `http://localhost:8080`，粘贴一个视频/音频 URL，发送
   「转写并入库」类型的问题
3. 观察回复：Agent 依次调用 `download_audio` → `transcribe_audio` →
   `index_transcript`，返回转写摘要与入库结果
4. 追问视频内容细节；应命中 `search_knowledge` 检索并给出带来源的答案
5. 重启 gateway 后携带同一 `session_id` 继续对话（配置了 MySQL 时历史仍在）

## 配置

完整见 `config.example.yaml`：

| 配置段 | 说明 |
|--------|------|
| `server` | 网关监听地址（write_timeout 保持 0，转写/入库耗时较长） |
| `download` | yt-dlp cookies、媒体沙箱目录 |
| `asr` | ASR 服务地址、模型 provider 与转写参数 |
| `embedding` | Embedding 服务地址、provider、维度、批大小 |
| `llm` | openai / ollama / deepseek |
| `qdrant` | 向量库地址、集合名、向量维度 |
| `rag` | 切分 rune 数、重叠、默认 top_k |
| `agent` | ReAct 最大步数、历史条数上限 |
| `mysql` | 会话持久化 DSN（留空则用内存存储） |

## 项目结构

```
vidwise/
├── main.go                  # 入口：装配 LLM/ASR/Embedding/Qdrant/Agent/Gin，优雅退出
├── cmd/download/            # yt-dlp / ffmpeg CLI 封装
├── internal/
│   ├── agent/               # eino ReAct Agent（系统提示词、max steps）
│   ├── appconfig/           # 配置加载与默认值/校验
│   ├── asr/                 # ASR 服务 HTTP 客户端
│   ├── chat/                # 会话 Service + HistoryStore（MySQL/内存两实现）
│   ├── embedding/           # Embedding 服务 HTTP 客户端
│   ├── healthcheck/         # /health 探测
│   ├── llm/                 # eino-ext 聊天模型工厂（openai/ollama/deepseek）
│   ├── rag/                 # rune 滑窗切分
│   ├── server/              # Gin 路由、/agent/chat handler、内嵌 Web UI
│   ├── storage/mysql/       # GORM 客户端
│   ├── tool/                # eino 工具：download_audio/transcribe_audio/
│   │                        #   index_transcript/search_knowledge + 路径/URL 校验
│   └── vectorstore/         # Qdrant gRPC 客户端（upsert/delete/search）
├── asr_service/             # Python ASR (FastAPI, :8001)
├── embedding_service/       # Python Embedding (FastAPI, :8003)
├── config.example.yaml
├── docker-compose.example.yaml
└── Makefile
```

## License

MIT