# VidWise 精简为最小可运行 MVP

## 1. 目标与范围

把 VidWise 收敛为一个能端到端跑通的 MVP：

> 用户在对话里给出视频链接 → **Eino ReAct Agent** 自主调用 tool：下载音频 → 调 ASR 服务转写 → 调 Embedding 服务向量化并写入 Qdrant → 之后的提问由 agent 调检索 tool 从 Qdrant 召回片段并回答。

已确认的决策：

| 项 | 决策 |
|---|---|
| 核心闭环 | 完整 RAG 闭环（下载 → ASR → 入库 → 检索问答） |
| 向量库 | 保留 Qdrant |
| MySQL | 保留，用于会话消息持久化；未配置 DSN 时退化为进程内内存存储 |
| 删除 | web search、MCP、video_summary、rerank、legacy（/extract、/format、knowledgeagent、chatagent、task、capability/manifest 等）、paragraph 格式化 |
| 交互 | Gin：`POST /agent/chat` + `GET /health` + 精简单页 Web UI |
| LLM | 保留 openai / ollama / deepseek 三种 eino-ext ChatModel |
| Python 服务 | 保留全部模型提供商接入；去掉 VAD、流式、分片优化、缓存、rerank 等优化逻辑，只保证可运行 |

不在范围内：多用户/鉴权、知识库按用户隔离、异步任务、流式输出、source 管理（列表/删除）、用户记忆。

## 2. 目标架构

```
浏览器 (index.html)
   │  POST /agent/chat {session_id?, message}
   ▼
Gin (:8080)  ──  chat.Service
                   ├─ HistoryStore (MySQL / Memory)
                   └─ Eino react.Agent (ToolCallingChatModel: openai|ollama|deepseek)
                         ├─ download_audio     → yt-dlp
                         ├─ transcribe_audio   → ASR Service (:8001) /transcribe
                         ├─ index_transcript   → Embedding Service (:8003) /embed → Qdrant upsert
                         └─ search_knowledge   → Embedding Service (:8003) /embed → Qdrant search
```

关于“embedding 封装为 tool”：直接暴露一个返回向量数组的 `embed_texts` tool 对 LLM 没有用（LLM 无法消费上千维浮点数，还会撑爆上下文）。因此 embedding 服务通过 `index_transcript` 和 `search_knowledge` 两个 tool 被 agent 调用，tool 内部调用 embedding 服务。ASR 服务对应 `transcribe_audio` tool。

转写文本不经过 LLM 中转：`transcribe_audio` 把全文写入工作目录下的 `.txt` 文件，只返回路径、时长、字数和前 500 字预览；`index_transcript` 读取该文件入库。这样避免 LLM 在 tool 参数里复述长文本导致截断或篡改。

## 3. Go 侧目标目录结构

```
main.go                       # 加载配置、构建依赖、启动 Gin（删除 worker 模式）
cmd/download/                 # 仅保留 Audio（删除 Video）
internal/
  appconfig/config.go         # 重写：只含 MVP 配置段
  asr/client.go               # 保留 HTTP 客户端（删除 tool.go）
  embedding/client.go         # 由 internal/model/embed.go 迁移（删除 rerank）
  llm/chatmodel.go            # 由 paragraph/chatmodel.go 迁移，返回 ToolCallingChatModel
  vectorstore/qdrant.go       # 由 storage/qdrant 精简：连接、建集合、Upsert、Search
  rag/chunk.go                # 按字符窗口切分（带重叠）
  tool/                       # 4 个 eino tool + 路径校验
  agent/agent.go              # 构建 eino react.Agent
  chat/                       # model.go、repo.go(MySQL)、memory.go、service.go
  server/                     # router.go、handler.go、web/index.html
  healthcheck/http.go         # 保留，ASR/Embedding 启动探活用
```

删除的目录/文件：`cmd/audio`、`cmd/video`、`cmd/worker`、`internal/{background,capability,chatagent,chunk,config,extractor,knowledgeagent,mcp,memory,model,paragraph,ragregistry,ragruntime,search,storage,task,tools,trace,user,video_summary,videoinput}`、`internal/rag` 现有全部文件、`internal/server` 现有全部文件、`internal/tool` 现有全部文件、`internal/agent` 现有全部文件、`main_test.go`（按新代码重写）、`video_summary_service/` 整个目录。

代码在 git 中（工作区干净，HEAD=`16ce928`），删除可通过 git 恢复。

## 4. 配置（`internal/appconfig/config.go` 重写）

Go 只解析自己用到的字段；`asr.model.*`、`embedding.provider` 等仍写在同一个 `config.yaml` 里，由 Python 服务读取（yaml.v3 默认忽略未知字段）。

```go
type Config struct {
    Server    ServerConfig    `yaml:"server"`    // addr, 默认 ":8080"
    Download  DownloadConfig  `yaml:"download"`  // cookies_path, work_dir(默认 ".vidwise/media")
    ASR       ASRConfig       `yaml:"asr"`       // base_url, timeout(默认 10m), language(默认 zh)
    Embedding EmbeddingConfig `yaml:"embedding"` // base_url, model, timeout(默认 2m)
    LLM       LLMConfig       `yaml:"llm"`       // provider, model, base_url, api_key, api_key_env, path, timeout, temperature, max_tokens, keep_alive
    Qdrant    QdrantConfig    `yaml:"qdrant"`    // host, port, collection, vector_dim
    RAG       RAGConfig       `yaml:"rag"`       // chunk_runes(800), overlap_runes(100), top_k(5)
    MySQL     MySQLConfig     `yaml:"mysql"`     // dsn(可空), max_open, max_idle
    Agent     AgentConfig     `yaml:"agent"`     // max_step(默认 20), history_limit(默认 20)
}
func Load(path string) (Config, error) // 读取 → 填默认值 → 校验（llm.provider/model 必填，qdrant.vector_dim>0）
```

`config.example.yaml` 重写为上述段落 + Python 需要的 `asr.model.*`、`asr.transcribe.{beam_size,initial_prompt,max_new_tokens,max_initial_prompt_tokens}`、`embedding.{provider,model,device,api_*,dimensions,batch_size}`。删除 `video_summary/search/rerank/mcp/upload/task/asr.stream/asr.transcribe.vad`。

## 5. Eino Tools（`internal/tool/`）

全部用 `utils.InferTool` 构造，返回 `tool.InvokableTool`，直接交给 react agent。删除原 `Wrapper`（重试+熔断）与 `Registry`，移除 `github.com/sony/gobreaker` 依赖。每个 tool 调用时 `slog.Info` 记录名称与耗时。

| Tool | 输入 | 输出 | 实现 |
|---|---|---|---|
| `download_audio` | `url`（必填） | `audio_path`, `title?` | 在 `work_dir` 下生成 `<uuid>` 作为 outputBase，调用 `download.Audio(url, base, cookies)` |
| `transcribe_audio` | `audio_path`（必填）, `language?` | `transcript_path`, `duration`, `chars`, `preview` | `asr.Client.Transcribe`；全文写入 `<audio_path 去扩展名>.txt` |
| `index_transcript` | `transcript_path`（必填）, `source_name`（必填）, `source_url?` | `source_id`, `chunks` | 读文件 → `rag.Split` → `embedding.Client.Embed`（按 16 条一批）→ `vectorstore.Upsert` |
| `search_knowledge` | `query`（必填）, `top_k?` | `results: [{text, source_name, source_url, score}]` | `Embed([query])` → `vectorstore.Search` |

安全边界（LLM 产生的参数属于不可信输入）：
- `download_audio` 只接受 `http://` / `https://` 开头的 URL；输出路径由服务端生成，不让 LLM 指定。yt-dlp 通过 `exec.Command` 参数数组调用，无 shell 拼接（沿用现状）。
- `transcribe_audio`、`index_transcript` 的路径参数必须位于 `work_dir` 内：`filepath.Abs` + `filepath.Rel` 校验，不允许 `..` 逃逸，否则返回错误。
- `top_k` 限制在 1–20。

`source_id` 用 `uuid.NewSHA1(namespace, source_url 或 transcript_path)` 生成；同一来源重复入库时先按 `source_id` 删除旧点再写入，避免重复。

## 6. 向量存储（`internal/vectorstore/qdrant.go`）

基于现有 `pb.PointsClient` / `pb.CollectionsClient`（gRPC），保留 insecure 连接方式：

```go
type Store struct{ conn *grpc.ClientConn; points pb.PointsClient; collections pb.CollectionsClient; collection string }
type Chunk struct{ Text, SourceID, SourceName, SourceURL string; Index int }
type Hit struct{ Chunk; Score float32 }

func New(ctx context.Context, addr, collection string, dim uint64) (*Store, error) // 连接 + 不存在则建集合(Cosine) + 给 source_id 建 keyword 索引
func (s *Store) ReplaceSource(ctx context.Context, sourceID string, chunks []Chunk, vectors [][]float32) error // Delete(filter source_id) → Upsert(wait=true)
func (s *Store) Search(ctx context.Context, vector []float32, topK uint64) ([]Hit, error)
func (s *Store) Close() error
```

point ID 用 `uuid.NewSHA1(sourceID, chunkIndex)`。Embedding 服务返回 `float64`，在写入前转为 `float32`。若返回向量维度与 `qdrant.vector_dim` 不一致，返回明确错误。

## 7. 切分（`internal/rag/chunk.go`）

```go
func Split(text string, size, overlap int) []string // 按 rune 滑窗；先 TrimSpace；空文本返回 nil；overlap>=size 时按 size/5 处理
```

## 8. Agent（`internal/agent/agent.go`）

```go
func New(ctx context.Context, cm model.ToolCallingChatModel, tools []tool.BaseTool, maxStep int) (*react.Agent, error) {
    return react.NewAgent(ctx, &react.AgentConfig{
        ToolCallingModel: cm,
        ToolsConfig:      compose.ToolsNodeConfig{Tools: tools},
        MessageModifier:  func(_ context.Context, in []*schema.Message) []*schema.Message {
            return append([]*schema.Message{schema.SystemMessage(SystemPrompt)}, in...)
        },
        MaxStep: maxStep,
    })
}
```

`SystemPrompt`（中文）要点：
1. 用户给出视频链接（或含链接的分享文案）时，依次调用 `download_audio` → `transcribe_audio` → `index_transcript`，然后基于预览简要告知用户视频已入库及内容概要。
2. 用户提问视频相关内容时，先调用 `search_knowledge`，只依据检索结果回答，并标注来源名称；检索为空时如实说明。
3. 不要编造文件路径，路径只能来自之前 tool 的返回值。

`internal/llm/chatmodel.go`：沿用现有三种 provider 的构造逻辑，返回类型改为 `model.ToolCallingChatModel`（已确认 openai v0.1.13、ollama v0.1.9、deepseek v0.1.6 的 `*ChatModel` 都实现了 `WithTools`）。删除 `disable_sampling_params` 自动探测等逻辑，只保留显式字段。

## 9. 会话（`internal/chat/`）

```go
type HistoryStore interface {
    CreateSession(ctx context.Context, title string) (string, error)
    SessionExists(ctx context.Context, id string) (bool, error)
    AddMessage(ctx context.Context, sessionID, role, content string) error
    RecentMessages(ctx context.Context, sessionID string, limit int) ([]Message, error)
}
```
- `repo.go`：保留现有 GORM `Session`/`Message` 模型与 `AutoMigrate`，实现上述接口（删除 `ListSessionsByUser`、`GenerateTitle` 等未用方法）。
- `memory.go`：`sync.Mutex` + map 的内存实现。
- `service.go`：

```go
func (s *Service) Chat(ctx context.Context, sessionID, message string) (sessionID string, reply string, err error)
// 1. sessionID 为空 → CreateSession(标题取 message 前 30 字)；不存在 → 返回 ErrSessionNotFound
// 2. history := RecentMessages(limit)，转成 schema.Message（只含 user/assistant 文本）
// 3. out, err := agent.Generate(ctx, append(history, schema.UserMessage(message)))
// 4. 成功后依次保存 user、assistant 两条消息；失败时不保存，返回错误
```
只持久化用户与最终回复文本，中间 tool 调用消息不入库（MVP 简化，后续轮次靠 `search_knowledge` 取知识）。

## 10. HTTP（`internal/server/`）

| 路由 | 说明 |
|---|---|
| `GET /health` | `{"status":"ok"}` |
| `POST /agent/chat` | 请求 `{"session_id":"可选","message":"必填"}`；响应 `{"session_id","reply"}`；message 为空 → 400；会话不存在 → 404；agent 失败 → 500 `{"error"}` |
| `GET /` | `//go:embed web/index.html` 单页：消息列表 + 输入框，`session_id` 存 `localStorage`，"新对话"按钮清空 |

`http.Server`：`ReadHeaderTimeout=5s`，`WriteTimeout=0`（下载+转写可能持续数分钟，请求同步阻塞），SIGINT/SIGTERM 优雅退出（沿用 main.go 现有逻辑）。

## 11. 启动流程（`main.go`）

1. `appconfig.Load("config.yaml")`，失败 panic。
2. `llm.NewChatModel` —— 失败直接退出（agent 必需）。
3. `asr.NewClient` + `Health`；`embedding.NewClient` + `Health`；`vectorstore.New`。三者任一失败都 **退出并打印原因**：MVP 的闭环依赖它们，静默降级只会让 agent 在调用时才报错，更难排查。
4. MySQL：`dsn` 非空则连接 + AutoMigrate，失败退出；`dsn` 为空则使用内存存储并 `slog.Warn`。
5. 构建 4 个 tool → `agent.New` → `chat.NewService` → `server.Router` → 监听。

删除 `-mode worker`。

## 12. Python 服务精简

HTTP 契约保持不变，Go 客户端无需适配：
- ASR `POST /transcribe`（multipart：`file`、`language`、`beam_size`、`vad_filter`、`initial_prompt`）→ `{text, language, language_probability, duration, segments[{start,end,text}]}`；`vad_filter` 仍接受但忽略。
- Embedding `POST /embed` `{texts, model}` → `{embeddings}`；`GET /health` → `{"status":"ok"}`。

### 12.1 `asr_service/`
保留全部 provider：whisper、faster-whisper、aliyun、xfyun、baidu（含别名）。

删除（优化逻辑）：
- `WS /stream` 及其全部辅助函数（`_decode_pcm`、`_event_offsets`、`_create_vad_iterator`、`_finalize_stream_utterance`、`_run_transcribe`、`_transcribe_audio`），`stream_client.py`，`asr.stream` 配置。
- Silero VAD 加载、`WhisperBackend._apply_vad`、faster-whisper `vad_parameters`、`asr.transcribe.vad` 配置。
- 所有 `np.ndarray` / `http://` / `oss://` 输入分支及 `_ndarray_*` 函数（只剩 `/transcribe` 传入文件路径一条路径）。
- whisper 窗口 2s 重叠、`no_repeat_ngram_size`；faster-whisper `cpu_threads/num_workers/condition_on_previous_text/temperature` 调优参数。
- xfyun 分片上传（`_transcribe_file_chunks`、`_split_audio_file`、`xfyun_chunk_seconds`）与静音兜底（`_empty_transcription_result`、`XFYUN_SILENT_FAIL_TYPE`）；`XFYunTranscriptionError` 简化为 `RuntimeError`。
- Baidu access token 过期缓存（每次请求重新取 token）。
- 请求日志中间件与自定义异常处理器。

保留（硬约束，不是优化）：
- whisper(transformers) 25s 窗口切分：模型一次只处理 30s，不切分会截断长音频。
- whisper prompt/`max_new_tokens` 预算：解码器上限 448 token。
- Baidu ffmpeg 转 16kHz 单声道 WAV 并按 ≤60s 分段：百度短语音接口限制。
- xfyun 上传 + 轮询 + HMAC 签名；Aliyun base64 大小上限检查。

`requirements.txt`：移除 `silero-vad`（及仅流式使用的依赖，实施时按 import 实际情况确定）。`test_backends.py`：删除针对已移除功能的用例，其余保留并保证通过。

### 12.2 `embedding_service/`
保留 provider：local（sentence-transformers：qwen/bge/bge-large/本地路径）、aliyun、siliconflow。
删除：`/rerank` 端点、所有 Rerank backend/别名表/配置读取、`/health` 中的 rerank 字段、懒加载单例（改为模块加载时创建 backend）。`test_backends.py` 同步删除 rerank 用例。

### 12.3 `video_summary_service/`
整个目录删除。

## 13. 其他文件

- `go.mod`：`go mod tidy` 后移除 gobreaker、mcp-go、gomarkdown 等不再使用的依赖。
- `Makefile`：保留 `deps`、`deps-python`、`deps-embedding`、`run`、`run-asr`、`run-embedding`、`run-all-bg`、`stop-all`、`test`；删除 video summary、worker、db-migrate 及 8082 端口。
- `docker-compose.example.yaml`：保留 MySQL + Qdrant，更新注释。
- `README.md`、`CLAUDE.md`、`AGENTS.md`：按新架构重写（当前内容已过时，保留会误导后续开发）。

## 14. 测试

- `internal/rag`：`Split` 的空文本、短文本、重叠边界。
- `internal/tool`：路径校验（work_dir 内/外、`..` 逃逸）、URL 校验；`transcribe_audio`/`index_transcript`/`search_knowledge` 用 `httptest` 假 ASR/Embedding 服务 + 接口化的向量存储假实现。
- `internal/chat`：内存存储 + 假 agent（`Generator` 接口）验证会话创建、历史拼接、失败不落库。
- `internal/server`：`/agent/chat` 400/404/200。
- `internal/asr`、`internal/healthcheck` 现有测试保留。
- 验证：`go build ./... && go vet ./... && go test ./...`；Python `python -m pytest asr_service embedding_service`（若环境已装依赖）或至少 `python -m py_compile`。

为便于测试，tool 依赖接口而非具体类型：

```go
type Transcriber interface{ Transcribe(ctx context.Context, path, lang string) (asr.TranscribeResponse, error) }
type Embedder    interface{ Embed(ctx context.Context, texts []string) ([][]float64, error) }
type VectorStore interface{ ReplaceSource(...) error; Search(...) ([]vectorstore.Hit, error) }
```

## 15. 数据流

```
"帮我看看 https://b23.tv/xxx 讲了什么"
→ POST /agent/chat → chat.Service.Chat → react.Agent.Generate
→ LLM tool_call download_audio{url}        → .vidwise/media/<uuid>.mp3
→ LLM tool_call transcribe_audio{path}     → ASR /transcribe → <uuid>.txt + preview
→ LLM tool_call index_transcript{path,name}→ Split → /embed → Qdrant ReplaceSource
→ LLM 最终回复（概要） → 保存 user/assistant 消息 → {session_id, reply}

"里面提到的第二个方法是什么？"
→ LLM tool_call search_knowledge{query} → /embed → Qdrant Search → top_k 片段
→ LLM 依据片段回答
```

## 16. 边界与异常

| 场景 | 处理 |
|---|---|
| yt-dlp 失败 | tool 返回错误（含 yt-dlp 输出末尾 500 字）；eino ToolsNode 把错误返回给 agent，agent 失败时 HTTP 500 |
| ASR 返回空文本 | `transcribe_audio` 返回错误 "transcript is empty" |
| 向量维度不符 | `index_transcript` 返回错误，提示检查 `qdrant.vector_dim` 与 embedding 模型 |
| 知识库为空 | `search_knowledge` 返回空 `results`，由 agent 告知用户 |
| 超过 MaxStep | react 返回错误 → 500 |
| 历史中 tool 消息 | 不入库，只回放 user/assistant 文本 |

注意：eino 默认会把 tool 返回的 error 直接作为整个 Generate 的错误。为让 agent 能感知错误并向用户解释，tool 内部把业务错误包装为正常返回 `{"error": "..."}`（实施时验证 v0.9.0 的 `ToolsNodeConfig` 是否有更合适的错误处理选项，若有则优先使用）。

## 17. 预期结果

- `go build ./...`、`go vet ./...`、`go test ./...` 通过。
- 启动 Qdrant（docker）+ `make run-asr` + `make run-embedding` + `make run` 后，浏览器打开 `http://localhost:8080`，发送视频链接即可完成下载→转写→入库，随后可就视频内容多轮问答。
- Go 侧代码量从约 1.5 万行（含测试）降到约 1500 行以内；Python 两个服务同样明显缩减，且所有 provider 仍可配置使用。
