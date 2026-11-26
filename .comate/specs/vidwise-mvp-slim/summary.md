# VidWise MVP 精简 — 实施总结

## 目标

将 VidWise 从功能繁多的平台精简为**最小可运行 MVP**：保留 CloudWeGo eino ReAct Agent
框架，ASR 与 Embedding 仍为独立 Python 服务（:8001 / :8003），通过 HTTP 被封为 eino
工具由 Agent 调用，形成完整 RAG 闭环：视频 URL → 下载音频 → 转写 → 向量化入库 →
基于知识库问答。移除了 Web 搜索、MCP、视频摘要、rerank、段落格式化、异步任务、
能力清单等子系统。

## 交付内容

### Go 网关（重写/新建）

| 模块 | 文件 | 说明 |
|------|------|------|
| 入口 | `main.go` | 装配 LLM/ASR/Embedding/Qdrant（失败即退出）、MySQL 可选、ReAct Agent、优雅退出 |
| 配置 | `internal/appconfig/` | MVP 配置结构体、默认值、校验（llm.model 必填、vector_dim>0、overlap<chunk） |
| LLM | `internal/llm/chatmodel.go` | openai / ollama / deepseek 三分支，返回 `einomodel.ToolCallingChatModel` |
| Agent | `internal/agent/agent.go` | `react.NewAgent` + 中文系统提示词（上传链路 / 检索优先 / 禁止编造路径 / 来源标注） |
| 工具 | `internal/tool/` | `download_audio`、`transcribe_audio`、`index_transcript`、`search_knowledge`；路径沙箱与 URL 校验 |
| ASR 客户端 | `internal/asr/client.go` | multipart `/transcribe` 调用 |
| Embedding 客户端 | `internal/embedding/client.go` | JSON `/embed` 调用，数量一致性校验 |
| 向量库 | `internal/vectorstore/qdrant.go` | Qdrant gRPC、Cosine、按 source_id 替换、sha256→uuid |
| 切分 | `internal/rag/chunk.go` | rune 滑窗 + 重叠 + 空白归一化 |
| 会话 | `internal/chat/` | Service + HistoryStore 接口；MySQL(GORM) 与内存两实现；仅成功时持久化 |
| HTTP | `internal/server/` | `GET /health`、`POST /agent/chat`（400/404/500 映射）、`GET /` 精简单页 UI |
| 基建 | `config.example.yaml`、`Makefile`、`docker-compose.example.yaml` | MVP 版配置与脚本（MySQL + Qdrant） |

### Python 服务（精简）

- **asr_service**：删除 `/stream` WebSocket、Silero VAD、日志中间件/异常处理器；
  `backends.py` 删除 ndarray/URL 分支、`_apply_vad`、whisper 2s 重叠与 ngram 抑制、
  faster-whisper 调优参数、讯飞分片与静音兜底（错误改为带描述 RuntimeError）、
  百度 token 缓存（每次请求重新获取）；保留 whisper 25s 窗口与 prompt 预算、
  百度 ffmpeg 16kHz 分片、讯飞上传+轮询+HMAC 签名、阿里云大小检查；统一
  `transcribe(audio, *, language, beam_size, initial_prompt)` 签名；
  删除 `stream_client.py`；`requirements.txt` 移除 websockets。
- **embedding_service**：删除 `/rerank`、RerankBackend/CrossEncoder/Cosine/HTTP rerank
  全部代码与配置读取、懒加载改为 startup 创建；保留 local/aliyun/siliconflow 三种
  embedding provider。

### 测试与文档

- Go：`go build ./... && go vet ./... && go test ./...` 全部通过（含 agent 假模型
  工具调用流转、工具校验、chat 服务、handler、配置、切分等测试）。
- Python：全部 6 个文件 `python -m py_compile` 通过（单元测试用例已同步裁剪；
  Python 测试需 ≥3.10 环境运行，本机仅有 3.9.6，故未实际执行，开发环境 pycache
  显示为 3.14）。
- 重写 `README.md`（含端到端冒烟步骤）、`CLAUDE.md`、`AGENTS.md`。

## 验证结果

```
go build ./...  ✔
go vet ./...    ✔
go test ./...   ✔ (14 个包全部 ok)
py_compile      ✔ (asr_service 3 文件 + embedding_service 3 文件)
```

全仓 grep 确认无残留的 `video_summary`/`/extract`/`capabilities`/`mcp` 等旧子系统引用
（文档中仅作为「已移除」说明出现）。

## 端到端冒烟（文档化，见 README）

1. `make run-all-bg`（或分别 `make run-embedding` / `make run-asr` / `make run`），
   提前启动 Qdrant（`docker compose -f docker-compose.example.yaml up -d`）。
2. 浏览器打开 `http://localhost:8080`，粘贴视频 URL 发送转写入库请求。
3. Agent 依次调用 `download_audio → transcribe_audio → index_transcript`。
4. 追问细节，命中 `search_knowledge`，回复带来源。
5. 配置 MySQL 时重启后携带同一 `session_id` 可继续对话。

## 注意事项

- 转写为同步长耗时操作：`server.write_timeout` 保持 0，Agent `max_step` 默认 20。
- ASR 适配器 `/transcribe` 仍接受 `vad_filter` 表单字段以兼容 Go 客户端，MVP 后端忽略之。
- 视频源 URL 仅允许 http/https；工具只读写 `download.work_dir` 沙箱内的文件。
- 内存会话存储重启即失；要持久化请在 `mysql.dsn` 配置 MySQL。