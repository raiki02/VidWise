# VidWise MVP 精简任务计划

> 依据 `.comate/specs/vidwise-mvp-slim/doc.md`。所有 Go 改动以 `go build ./... && go vet ./... && go test ./...` 通过为验收标准；连续实施，顺序执行。

- [x] Task 1: 保留基础包、删除废弃包及文件
    - 1.1: 删除：`cmd/audio`、`cmd/video`、`cmd/worker`、`internal/{background,capability,chatagent,chunk,config,extractor,knowledgeagent,mcp,memory,model,paragraph,ragregistry,ragruntime,search,storage,task,tools,trace,user,video_summary,videoinput}` 目录
    - 1.2: 删除：`internal/rag`、`internal/server`、`internal/tool`、`internal/agent`、`internal/mcp` 现有文件，以及 `video_summary_service/`、`main_test.go`
    - 1.3: 删除 go.mod 中不再使用的依赖并 `go mod tidy`（此时编译必然失败，先只做清理，任务完成后统一修复）

- [x] Task 2: 重写 `internal/appconfig/config.go`（配置加载与默认值）
    - 2.1: 定义 MVP 配置结构体（server/download/asr/embedding/llm/qdrant/rag/mysql/agent）
    - 2.2: `Load` 加载 YAML、填充默认值、校验必填项（llm.provider/model、qdrant.vector_dim>0）
    - 2.3: 保留 `LLMConfig.ResolvedAPIKey`、`TimeoutDuration`、`KeepAliveDuration` 等最小必要方法
    - 2.4: 删除 `config_test.go` 中针对已删配置段的用例，保留并适配剩余断言

- [x] Task 3: 迁移并精简 LLM ChatModel
    - 3.1: 新建 `internal/llm/chatmodel.go`：openai/ollama/deepseek 三分支，返回 `model.ToolCallingChatModel`
    - 3.2: 从 `internal/paragraph/chatmodel.go` 迁移构造逻辑（APIKey、BaseURL、Timeout、Temperature、MaxTokens、KeepAlive、Path）；删除 `disable_sampling_params` 探测
    - 3.3: 删除 `internal/paragraph`（若 Task 1 已删则跳过）

- [x] Task 4: 精简 ASR 客户端与 Embedding 客户端
    - 4.1: `internal/asr/client.go`：保留 `Transcribe`/`Health`/`TranscribeResponse`，删除 `TranscribeOptions` 未用字段（若被 transcribe 参数使用则保留并传递）
    - 4.2: 删除 `internal/asr/tool.go`，保留 `internal/asr/client_test.go`
    - 4.3: 新建 `internal/embedding/client.go`（由 `internal/model/embed.go` 迁移）：`NewClient`/`Health`/`Embed`/`EmbedResponse`；删除 `RerankClient`
    - 4.4: 删除 `internal/model` 目录

- [x] Task 5: 向量存储 `internal/vectorstore/qdrant.go`
    - 5.1: `New(ctx, addr, collection, dim)`：gRPC 连接（insecure）+ 不存在则创建 Cosine 集合（沿用 pb API）
    - 5.2: `ReplaceSource`：按 source_id 删除旧点 → 批量 Upsert（wait=true），point ID 用 uuid.NewSHA1
    - 5.3: `Search`：向量检索、返回 text/source/score
    - 5.4: 用 `vectorStore` 接口面向上层，提供测试假实现

- [x] Task 6: 文本切分 `internal/rag/chunk.go`
    - 6.1: `Split(text string, size, overlap int) []string`（rune 滑窗、TrimSpace、空文本返回 nil、overlap 边界处理）
    - 6.2: 单元测试：空文本/短文本/重叠/归一化

- [x] Task 7: 构建 Eino Tools（`internal/tool/`）
    - 7.1: `base.go`：路径校验（work_dir 内、禁止 `..` 逃逸）、URL 校验（仅 http/https）、`tool.NewDownloadTool` 等构造函数签名
    - 7.2: `audio.go`：`download_audio`（yt-dlp，outputBase 服务端生成 `<work_dir>/<uuid>`）
    - 7.3: `asr_tool.go`：`transcribe_audio`（调 Transcriber 接口，全文写 `<id>.txt`，返回 path/duration/chars/preview）
    - 7.4: `rag_index.go`：`index_transcript`（读文件 → Split → EmbeddingClient.Embed → VectorStore.ReplaceSource）
    - 7.5: `rag_query.go`：`search_knowledge`（Embed([query]) → Search，top_k 1–20）
    - 7.6: 单测：路径/URL 校验；用 httptest 假 ASR/Embedding + 假 VectorStore 覆盖三个工具

- [x] Task 8: Agent 构建 `internal/agent/agent.go`
    - 8.1: `New(ctx, cm, tools, maxStep)` 封装 `react.NewAgent`（MessageModifier 注入系统提示）
    - 8.2: 写系统提示词：视频链接→三步处理链；提问→先检索；禁止编造路径；来源标注
    - 8.3: 单测：假 ChatModel 验证 tool 调用流转（复用现有 react/chatmodel 测试思路）

- [x] Task 9: 会话 `internal/chat/`
    - 9.1: `HistoryStore` 接口 + `Message` 模型（保留 session/message GORM 模型与 AutoMigrate）
    - 9.2: `repo.go`：实现 HistoryStore（CreateSession/SessionExists/AddMessage/RecentMessages），删除未用方法
    - 9.3: `memory.go`：内存实现（Mutex + map）
    - 9.4: `service.go`：`Chat` 流程（建会话/取历史→Generate→存两条消息），错误语义（400/404/500 由 handler 层映射）
    - 9.5: 测试：内存存储 + 假 Generator 覆盖成功/失败/会话不存在

- [x] Task 10: HTTP `internal/server/`
    - 10.1: `router.go`：Gin 路由 `/health`、`POST /agent/chat`、`GET /`
    - 10.2: `handler.go`：参数校验与错误映射
    - 10.3: `web/index.html`：精简单页（消息列表、输入框、session_id localStorage、新对话按钮）
    - 10.4: `server.go`：`NewHTTPServer`（WriteTimeout=0）复用现有实现
    - 10.5: 测试：handler 400/404/200 与健康检查

- [x] Task 11: 重写 `main.go`
    - 11.1: 删除 worker 模式；LLM/ASR/Embedding/Qdrant 任一失败即退出
    - 11.2: 构建 tool、agent、chat.Service，启动 Gin 并优雅退出
    - 11.3: 重写 `main_test.go`（若有必要的纯函数断言则保留，否则删除）

- [x] Task 12: 配置示例与构建脚本
    - 12.1: 重写 `config.example.yaml`（server/download/asr/embedding/llm/qdrant/rag/mysql/agent；Python 需要的 asr.model.*、asr.transcribe、embedding.* 保留）
    - 12.2: 更新 `Makefile`（删 video-summary/worker/db-migrate，更新注释与端口）
    - 12.3: 更新 `docker-compose.example.yaml` 注释

- [x] Task 13: 精简 `asr_service/`
    - 13.1: `app.py`：删 `/stream` 及流式辅助函数、Silero VAD 加载、日志中间件/异常处理器、`asr.stream`/`transcribe.vad` 配置读取
    - 13.2: `backends.py`：删 ndarray/URL 分支、`_apply_vad`、whisper 重叠、faster-whisper 调优参数、xfyun 分片与静音兜底（简化异常）、Baidu token 缓存；保留 whisper 25s 窗口、prompt 预算、Baidu ffmpeg 分片、xfyun 轮询签名、Aliyun 大小检查
    - 13.3: 删除 `stream_client.py`；`requirements.txt` 移除不再使用的包
    - 13.4: `test_backends.py` 同步删除对应用例并保证通过

- [x] Task 14: 精简 `embedding_service/`
    - 14.1: `app.py`：删 `/rerank`、rerank 配置/请求模型、懒加载改为启动时创建
    - 14.2: `backends.py`：删 RerankBackend/CrossEncoder/Cosine/HTTP rerank 相关全部代码
    - 14.3: `test_backends.py` 同步删除 rerank 用例

- [x] Task 15: 全量验证与文档更新
    - 15.1: `go mod tidy` 后 `go build ./... && go vet ./... && go test ./...` 全部通过
    - 15.2: Python `python -m py_compile`（或 pytest，若环境依赖可用）通过
    - 15.3: 按新架构重写 `README.md`、`CLAUDE.md`、`AGENTS.md`
    - 15.4: 端到端冒烟（文档化）：Qdrant + ASR + Embedding + gateway 启动，浏览器发链接完成一轮入库与问答
