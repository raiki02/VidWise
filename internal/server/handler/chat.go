package handler

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/cloudwego/eino/schema"
	"github.com/gin-gonic/gin"
	"github.com/raiki02/vidwise/internal/appconfig"
	"github.com/raiki02/vidwise/internal/chat"
	"github.com/raiki02/vidwise/internal/memory"
	"github.com/raiki02/vidwise/internal/paragraph"
	"github.com/raiki02/vidwise/internal/rag"
)

type ChatHandler struct {
	repo      *chat.Repo
	memRepo   *memory.Repo
	retriever *rag.Retriever
	llmCfg    appconfig.LLMConfig
}

func NewChatHandler(repo *chat.Repo, memRepo *memory.Repo, retriever *rag.Retriever, llmCfg appconfig.LLMConfig) *ChatHandler {
	return &ChatHandler{repo: repo, memRepo: memRepo, retriever: retriever, llmCfg: llmCfg}
}

// ---- Request / Response types ----

type ChatQueryRequest struct {
	SessionID string `json:"session_id"`
	UserID    string `json:"user_id"` // optional, for cross-session memory
	Query     string `json:"query" binding:"required"`
}

type ChatChunk struct {
	Text      string  `json:"text"`
	Score     float64 `json:"score"`
	SessionID string  `json:"session_id,omitempty"`
	ChunkIdx  int64   `json:"chunk_idx,omitempty"`
}

type ChatQueryResponse struct {
	SessionID    string       `json:"session_id"`
	Answer       string       `json:"answer"`
	Chunks       []ChatChunk  `json:"chunks,omitempty"`
	RAGTriggered bool         `json:"rag_triggered"`
	RAGReason    string       `json:"rag_reason,omitempty"`
	Question     string       `json:"question"`
}

type SessionListItem struct {
	ID        string `json:"id"`
	UserID    string `json:"user_id,omitempty"`
	Title     string `json:"title"`
	UpdatedAt string `json:"updated_at"`
}

type SessionDetail struct {
	ID       string         `json:"id"`
	UserID   string         `json:"user_id,omitempty"`
	Title    string         `json:"title"`
	Messages []chat.Message `json:"messages"`
}

// ---- RAG Trigger Evaluation ----

type ragEvalResult struct {
	ShouldRetrieve bool   `json:"should_retrieve"`
	Reason         string `json:"reason"`
}

// evaluateRAGNeed asks the LLM whether the query genuinely requires
// knowledge base retrieval. This prevents hallucination-prone RAG when
// the user is just chatting or asking about general knowledge.
func (h *ChatHandler) evaluateRAGNeed(ctx context.Context, query string, recentHistory string, userFacts string) ragEvalResult {
	if h.llmCfg.Enabled == nil || !*h.llmCfg.Enabled || h.llmCfg.Model == "" {
		// No LLM available — always attempt retrieval as best-effort
		return ragEvalResult{ShouldRetrieve: true, Reason: "llm_unavailable"}
	}
	if h.retriever == nil {
		return ragEvalResult{ShouldRetrieve: false, Reason: "retriever_unavailable"}
	}

	cm, err := paragraph.NewChatModel(ctx, h.llmCfg)
	if err != nil {
		slog.Warn("chat.rag_eval_llm_failed", "err", err)
		return ragEvalResult{ShouldRetrieve: true, Reason: "llm_init_failed"}
	}

	evalPrompt := `你是一个RAG（检索增强生成）系统的查询评估器。你的任务是判断用户的最新问题是否需要从视频知识库中检索相关信息。

需要触发RAG的情况（回答"是"）：
1. 问题涉及视频的具体内容、转录文本、或之前上传到知识库的文档
2. 用户询问关于某个视频的话题、细节、或片段
3. 用户想知道知识库中是否包含某个主题
4. 用户要求引用、查找、搜索具体信息

不需要触发RAG的情况（回答"否"）：
1. 简单的寒暄和闲聊（"你好"、"今天天气怎么样"）
2. 纯技术咨询，不涉及知识库内容（"go语言的goroutine怎么用"）
3. 用户在谈论自己的情况（"我正在学go"——这是用户个人信息，不是对知识库的查询）
4. 用户对AI能力的询问（"你能做什么"、"你怎么工作的"）
5. 用户要求总结当前对话（不涉及外部知识）

重要：用户谈论自己的个人信息（如学习计划、偏好等）不需要触发RAG，因为这些不是对知识库的查询。`

	userPrompt := fmt.Sprintf("用户最新问题：%s\n\n请判断是否需要从视频知识库检索。返回JSON格式：{\"should_retrieve\": true/false, \"reason\": \"判断理由\"}", query)

	if recentHistory != "" {
		userPrompt = fmt.Sprintf("最近对话：\n%s\n\n%s", recentHistory, userPrompt)
	}

	msgs := []*schema.Message{
		schema.SystemMessage(evalPrompt),
		schema.UserMessage(userPrompt),
	}

	resp, genErr := cm.Generate(ctx, msgs)
	if genErr != nil || resp.Content == "" {
		slog.Warn("chat.rag_eval_gen_failed", "err", genErr)
		return ragEvalResult{ShouldRetrieve: true, Reason: "eval_gen_failed"}
	}

	var result ragEvalResult
	if err := extractJSON(resp.Content, &result); err != nil {
		slog.Warn("chat.rag_eval_parse_failed", "content", resp.Content, "err", err)
		// Default to retrieve on parse failure (conservative)
		return ragEvalResult{ShouldRetrieve: true, Reason: "eval_parse_failed"}
	}

	slog.Info("chat.rag_eval", "should_retrieve", result.ShouldRetrieve, "reason", result.Reason)
	return result
}

// ---- ChatQuery ----

func (h *ChatHandler) ChatQuery(c *gin.Context) {
	var req ChatQueryRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "query is required"})
		return
	}

	ctx := c.Request.Context()
	sessionID := req.SessionID

	// Auto-create session if none provided
	if sessionID == "" {
		title := "新对话"
		s, err := h.repo.CreateSessionForUser(ctx, req.UserID, title)
		if err != nil {
			slog.Error("chat.create_session_failed", "err", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "创建会话失败"})
			return
		}
		sessionID = s.ID
	}

	// Save user message
	if _, err := h.repo.AddMessage(ctx, sessionID, "user", req.Query); err != nil {
		slog.Error("chat.save_user_msg_failed", "err", err)
	}

	// Auto-title: generate on first user message
	msgs, _ := h.repo.GetMessages(ctx, sessionID, 0)
	userCount := 0
	for _, m := range msgs {
		if m.Role == "user" {
			userCount++
		}
	}
	if userCount == 1 {
		func() {
		tCtx, tCancel := genCtx()
		defer tCancel()
		h.autoGenerateTitle(tCtx, sessionID, req.Query)
	}()
	}

	// Load user memory facts for cross-session context
	userFacts := ""
	if h.memRepo != nil && req.UserID != "" {
		userFacts = h.memRepo.FormatForPrompt(ctx, req.UserID)
	}

	// Build recent history for context
	var recentMsgs []chat.Message
	if h.repo != nil && sessionID != "" {
		recentMsgs, _ = h.repo.GetRecentMessages(ctx, sessionID, 20)
	}
	recentHistory := buildHistoryText(recentMsgs)

	// Step 1: Evaluate whether RAG is needed
	eval := h.evaluateRAGNeed(ctx, req.Query, recentHistory, userFacts)

	// Step 2: RAG retrieval (only if triggered)
	var chunks []rag.RelevantChunk
	if eval.ShouldRetrieve && h.retriever != nil {
		var filter *rag.RetrieveFilter
		if req.UserID != "" {
			filter = &rag.RetrieveFilter{UserID: req.UserID}
		}
		retrievedChunks, err := h.retriever.Retrieve(ctx, req.Query, filter)
		if err != nil {
			slog.Error("chat.retrieve_failed", "err", err)
		}
		chunks = retrievedChunks
	}

	// Step 3: Build answer with LLM (includes RAG context if available, plus user memory)
	answer := h.buildAnswer(ctx, sessionID, chunks, req.Query, eval, userFacts)

	// Save assistant response
	if _, err := h.repo.AddMessage(ctx, sessionID, "assistant", answer); err != nil {
		slog.Error("chat.save_assistant_msg_failed", "err", err)
	}

	// Step 4: Asynchronously extract/update user memory facts
	if h.memRepo != nil && req.UserID != "" {
		func() {
		mCtx, mCancel := genCtx()
		defer mCancel()
		h.extractMemoryFacts(mCtx, req.UserID, sessionID, req.Query, answer, recentHistory)
	}()
	}

	outChunks := make([]ChatChunk, 0)
	for _, c := range chunks {
		outChunks = append(outChunks, ChatChunk{
			Text:      c.Text,
			Score:     c.Score,
			SessionID: c.SessionID,
			ChunkIdx:  c.ChunkIdx,
		})
	}

	c.JSON(http.StatusOK, ChatQueryResponse{
		SessionID:    sessionID,
		Answer:       answer,
		Chunks:       outChunks,
		RAGTriggered: eval.ShouldRetrieve,
		RAGReason:    eval.Reason,
		Question:     req.Query,
	})
}

func (h *ChatHandler) buildAnswer(ctx context.Context, sessionID string, chunks []rag.RelevantChunk, query string, eval ragEvalResult, userFacts string) string {
	// Build RAG context
	ragContext := ""
	hasRAGContext := false
	for i, c := range chunks {
		if i > 0 {
			ragContext += "\n---\n"
		}
		ragContext += c.Text
		hasRAGContext = true
	}

	// Fetch recent chat history
	var history []chat.Message
	if h.repo != nil && sessionID != "" {
		history, _ = h.repo.GetRecentMessages(ctx, sessionID, 30)
	}
	historyText := buildHistoryText(history)

	// Try LLM
	if h.llmCfg.Enabled != nil && *h.llmCfg.Enabled && h.llmCfg.Model != "" {
		cm, err := paragraph.NewChatModel(ctx, h.llmCfg)
		if err == nil {
			var msgs []*schema.Message

			// Build system prompt with appropriate instructions based on RAG status
			systemPrompt := buildSystemPrompt(hasRAGContext, userFacts)
			msgs = append(msgs, schema.SystemMessage(systemPrompt))

			// Build user prompt
			var userPrompt string
			if hasRAGContext {
				userPrompt = fmt.Sprintf("视频转录文本（来自知识库，请务必根据这些内容回答，并标注来源）：\n%s\n\n用户问题：%s", ragContext, query)
				if historyText != "" {
					userPrompt = fmt.Sprintf("对话历史：\n%s\n\n%s", historyText, userPrompt)
				}
				userPrompt += "\n\n请根据视频知识库内容和对话历史回答用户的问题。如果引用了知识库中的具体信息，请标注内容来源。"
			} else {
				userPrompt = fmt.Sprintf("用户问题：%s", query)
				if historyText != "" {
					userPrompt = fmt.Sprintf("对话历史：\n%s\n\n用户最新问题：%s", historyText, query)
				}
				if userFacts != "" {
					userPrompt += "\n\n请结合用户画像信息，自然地个性化你的回复。"
				}
			}
			msgs = append(msgs, schema.UserMessage(userPrompt))

			resp, genErr := cm.Generate(ctx, msgs)
			if genErr == nil && resp.Content != "" {
				return resp.Content
			}
			slog.Warn("chat.llm_gen_failed", "err", genErr)
		} else {
			slog.Warn("chat.llm_unavailable", "err", err)
		}
	}

	// Fallback
	if hasRAGContext {
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("基于 %d 个相关文本片段：\n\n", len(chunks)))
		for i, c := range chunks {
			if i >= 5 {
				break
			}
			sb.WriteString(fmt.Sprintf("**片段 %d**（相关度 %.2f）：\n%s\n\n", i+1, c.Score, c.Text))
		}
		return sb.String()
	}
	return "抱歉，我暂时无法处理你的请求。请稍后再试。"
}

func buildSystemPrompt(hasRAG bool, userFacts string) string {
	var sb strings.Builder

	sb.WriteString("你是视频知识库问答助手。")

	if userFacts != "" {
		sb.WriteString("\n\n")
		sb.WriteString(userFacts)
		sb.WriteString("\n请自然地利用这些用户信息来个性化你的回复，但不要直接复述它们。")
	}

	if hasRAG {
		sb.WriteString(`

要求：
1. 仅根据提供的上下文回答，不要编造信息。
2. 提取上下文中与问题最相关的内容，用自己的话重新组织。
3. 回答要具体、有细节，不要泛泛而谈。
4. 如果上下文信息不足，明确告知用户，不要猜测。
5. 结合对话历史理解用户的意图和上下文。
6. 用中文回答，语言清晰易懂。
7. 当引用了知识库中某段内容时，注明"根据知识库中的内容..."以便用户追溯来源。`)
	} else {
		sb.WriteString(`

要求：
1. 准确回答用户问题，不要编造信息。
2. 如果问题超出你的知识范围，明确告知用户。
3. 结合对话历史理解用户的意图和上下文。
4. 用中文回答，语言清晰易懂。`)
	}

	return sb.String()
}

// ---- Memory Extraction ----

// extractMemoryFacts runs asynchronously after each turn to extract
// or update facts about the user from the conversation.
func (h *ChatHandler) extractMemoryFacts(ctx context.Context, userID, sessionID, userMsg, assistantReply, recentHistory string) {
	if h.llmCfg.Enabled == nil || !*h.llmCfg.Enabled || h.llmCfg.Model == "" {
		return
	}

	cm, err := paragraph.NewChatModel(ctx, h.llmCfg)
	if err != nil {
		slog.Warn("memory.extract_llm_failed", "err", err)
		return
	}

	// Get existing facts as context for the LLM
	existingJSON := "[]"
	if h.memRepo != nil {
		existingJSON = h.memRepo.DumpForPromptJSON(ctx, userID)
	}

	systemPrompt := `你是一个用户画像提取助手。你的任务是从用户与AI的对话中，提取关于用户的重要信息作为跨会话记忆。

提取规则：
1. 从用户的消息中提取用户的个人信息、偏好、状态、计划、技能等
2. 信息必须是用户明确表达或强烈暗示的，不要无中生有
3. 每条信息需要分类（category），例如：学习、工作、兴趣、偏好、目标、技能、个人情况等
4. 为每条信息设置置信度：
   - "high": 用户明确陈述（如"我正在学go"）
   - "medium": 从对话中合理推断（如多次问go相关问题，推断用户在学习go）
   - "low": 弱信号，仅作参考

动作（action）说明：
- "create": 这是一个新发现的信息，之前没有记录
- "update": 更新已有信息（用户提供了更具体的内容）
- "confirm": 用户在对话中再次提及，确认该信息仍然有效
- "supersede": 用户的新表达与旧信息矛盾，新信息更可靠（如用户先说"学go"，后来说"改用rust了"）

矛盾处理规则（重要）：
1. 当新旧事实矛盾时，必须判断哪个更可靠：
   - 用户的直接最新陈述 > 旧的直接陈述
   - 被多次提及的事实 > 只提过一次的事实
   - 有具体上下文支撑的事实 > 孤立陈述
   - 用户明确更正 > 推断
2. 不要"无脑去旧留新"——如果旧事实被多次确认，而新表述可能是一时口误，不应该覆盖
3. 不要"无脑去旧留新"——如果用户明确表示情况发生了变化，应该用新事实替代旧事实
4. 在reason字段中说明判断依据

输出格式：JSON数组
[
  {
    "category": "学习",
    "key": "编程语言",
    "value": "Go语言",
    "evidence": "用户说'我正在学go'",
    "source": "explicit",
    "confidence": "high",
    "action": "create",
    "target_id": "",
    "reason": ""
  }
]

注意：
- 只提取用户的信息，不提取AI的回复内容
- 不要提取临时性的、无意义的对话内容
- 如果用户只是打招呼或闲聊，返回空数组 []
- 如果新信息与已有信息完全相同，用"confirm"动作即可，不需要重复创建
- category用中文，key和value用中文简洁描述
- key应该是一个简短的属性名（如"正在学习的语言"），value是具体值（如"Go语言"）`

	userPrompt := fmt.Sprintf(
		"已有用户画像: %s\n\n最近对话:\n%s\n用户: %s\n助手: %s\n\n请从以上对话中提取关于用户的新信息或更新已有信息。",
		existingJSON, recentHistory, userMsg, truncateStr(assistantReply, 500),
	)

	msgs := []*schema.Message{
		schema.SystemMessage(systemPrompt),
		schema.UserMessage(userPrompt),
	}

	resp, genErr := cm.Generate(ctx, msgs)
	if genErr != nil || resp.Content == "" {
		slog.Warn("memory.extract_gen_failed", "err", genErr)
		return
	}

	var facts []memory.ExtractedFact
	if err := extractJSON(resp.Content, &facts); err != nil {
		slog.Warn("memory.extract_parse_failed", "content", resp.Content, "err", err)
		return
	}

	if len(facts) == 0 {
		slog.Info("memory.no_facts_extracted", "user_id", userID)
		return
	}

	// Apply extracted facts
	for _, f := range facts {
		switch f.Action {
		case "create":
			if _, err := h.memRepo.UpsertFact(ctx, userID, f, sessionID); err != nil {
				slog.Error("memory.create_failed", "err", err)
			}
		case "update":
			created, err := h.memRepo.UpsertFact(ctx, userID, f, sessionID)
			if err != nil {
				slog.Error("memory.update_failed", "err", err)
			} else {
				slog.Info("memory.fact_updated", "id", created.ID, "key", f.Key)
			}
		case "supersede":
			if f.TargetID != "" {
				// Create the new fact first, then supersede the old one
				newFact, err := h.memRepo.UpsertFact(ctx, userID, f, sessionID)
				if err != nil {
					slog.Error("memory.supersede_create_failed", "err", err)
					continue
				}
				if err := h.memRepo.SupersedeFact(ctx, userID, f.TargetID, newFact.ID, f.Reason); err != nil {
					slog.Error("memory.supersede_failed", "err", err)
				} else {
					slog.Info("memory.fact_superseded", "old", f.TargetID, "new", newFact.ID, "reason", f.Reason)
				}
			}
		case "confirm":
			if f.TargetID != "" {
				if err := h.memRepo.ConfirmFact(ctx, userID, f.TargetID); err != nil {
					slog.Error("memory.confirm_failed", "err", err)
				}
			}
		default:
			slog.Warn("memory.unknown_action", "action", f.Action)
		}
	}

	slog.Info("memory.extraction_done", "user_id", userID, "facts", len(facts))
}

// ---- Sessions ----

func (h *ChatHandler) ListSessions(c *gin.Context) {
	userID := c.Query("user_id")

	var sessions []chat.Session
	var err error

	if userID != "" {
		sessions, err = h.repo.ListSessionsByUser(c.Request.Context(), userID, 50)
	} else {
		sessions, err = h.repo.ListSessions(c.Request.Context(), 50)
	}

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取会话列表失败"})
		return
	}
	list := make([]SessionListItem, 0, len(sessions))
	for _, s := range sessions {
		list = append(list, SessionListItem{
			ID:        s.ID,
			UserID:    s.UserID,
			Title:     s.Title,
			UpdatedAt: s.UpdatedAt.Format("2006-01-02 15:04"),
		})
	}
	c.JSON(http.StatusOK, gin.H{"sessions": list})
}

func (h *ChatHandler) GetSession(c *gin.Context) {
	sessionID := c.Param("id")
	if sessionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "session id is required"})
		return
	}

	session, err := h.repo.GetSession(c.Request.Context(), sessionID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "会话不存在"})
		return
	}

	messages, _ := h.repo.GetMessages(c.Request.Context(), sessionID, 200)

	c.JSON(http.StatusOK, SessionDetail{
		ID:       session.ID,
		UserID:   session.UserID,
		Title:    session.Title,
		Messages: messages,
	})
}

func (h *ChatHandler) NewSession(c *gin.Context) {
	var req struct {
		UserID string `json:"user_id"`
	}
	_ = c.ShouldBindJSON(&req)

	s, err := h.repo.CreateSessionForUser(c.Request.Context(), req.UserID, "新对话")
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "创建会话失败"})
		return
	}
	c.JSON(http.StatusCreated, gin.H{
		"session_id": s.ID,
		"user_id":    s.UserID,
		"title":      s.Title,
	})
}

func (h *ChatHandler) RAGHealth(c *gin.Context) {
	status := "available"
	if h.retriever == nil {
		status = "unavailable"
	}
	memStatus := "unavailable"
	if h.memRepo != nil {
		memStatus = "available"
	}
	c.JSON(http.StatusOK, gin.H{
		"rag":    status,
		"llm":    h.llmCfg.Enabled,
		"memory": memStatus,
	})
}

// ---- User Memory endpoints ----

type MemoryFactResponse struct {
	ID         string `json:"id"`
	Category   string `json:"category"`
	Key        string `json:"key"`
	Value      string `json:"value"`
	Confidence string `json:"confidence"`
	Source     string `json:"source"`
	CreatedAt  string `json:"created_at"`
}

func (h *ChatHandler) GetUserFacts(c *gin.Context) {
	userID := c.Query("user_id")
	if userID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "user_id is required"})
		return
	}
	if h.memRepo == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "memory service unavailable"})
		return
	}

	facts, err := h.memRepo.GetFactsByUser(c.Request.Context(), userID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取用户画像失败"})
		return
	}

	list := make([]MemoryFactResponse, 0, len(facts))
	for _, f := range facts {
		list = append(list, MemoryFactResponse{
			ID:         f.ID,
			Category:   f.Category,
			Key:        f.Key,
			Value:      f.Value,
			Confidence: string(f.Confidence),
			Source:     string(f.Source),
			CreatedAt:  f.CreatedAt.Format("2006-01-02 15:04"),
		})
	}

	c.JSON(http.StatusOK, gin.H{"facts": list, "count": len(list)})
}

// ---- Helpers ----

func (h *ChatHandler) autoGenerateTitle(ctx context.Context, sessionID, firstQuery string) {
	title := firstQuery
	runes := []rune(title)
	if len(runes) > 50 {
		title = string(runes[:50])
	}
	title = strings.TrimSpace(title)
	if title == "" {
		title = "新对话"
	}
	if err := h.repo.UpdateSessionTitle(ctx, sessionID, title); err != nil {
		slog.Warn("chat.auto_title_failed", "err", err)
	} else {
		slog.Info("chat.auto_title", "session_id", sessionID, "title", title)
	}
}

func buildHistoryText(msgs []chat.Message) string {
	var sb strings.Builder
	for _, m := range msgs {
		if m.Role == "user" {
			sb.WriteString(fmt.Sprintf("用户: %s\n", m.Content))
		} else {
			sb.WriteString(fmt.Sprintf("助手: %s\n", m.Content))
		}
	}
	return sb.String()
}

// extractJSON tries to find and parse a JSON array or object from LLM output.
// It uses bracket-balancing to find the correct outermost JSON span,
// handling cases where LLM output contains JSON embedded in explanatory text.
func extractJSON(content string, v interface{}) error {
	content = strings.TrimSpace(content)

	// Try direct parse first
	if err := json.Unmarshal([]byte(content), v); err == nil {
		return nil
	}

	// Find the outermost balanced JSON object or array.
	// Track bracket depth so we don't get fooled by nested brackets or
	// stray braces in surrounding text.
	start := -1
	depth := 0
	var openBracket, closeBracket rune

	for i, ch := range content {
		if start == -1 {
			if ch == '[' || ch == '{' {
				start = i
				depth = 1
				if ch == '[' {
					openBracket, closeBracket = '[', ']'
				} else {
					openBracket, closeBracket = '{', '}'
				}
			}
			continue
		}

		if ch == openBracket {
			depth++
		} else if ch == closeBracket {
			depth--
			if depth == 0 {
				// Found matching closing bracket
				return json.Unmarshal([]byte(content[start:i+1]), v)
			}
		}
		// Handle nested opposite bracket types (e.g., { inside [ or [ inside {)
		if openBracket == '[' && ch == '{' {
			depth++
		} else if openBracket == '[' && ch == '}' {
			depth--
		} else if openBracket == '{' && ch == '[' {
			depth++
		} else if openBracket == '{' && ch == ']' {
			depth--
		}
	}

	// If we started but never closed, try to parse what we have
	if start >= 0 && start < len(content) {
		return json.Unmarshal([]byte(content[start:]), v)
	}

	return fmt.Errorf("no JSON found in: %s", truncateStr(content, 200))
}

func truncateStr(s string, maxRunes int) string {
	runes := []rune(s)
	if len(runes) <= maxRunes {
		return s
	}
	return string(runes[:maxRunes]) + "..."
}

// genCtx creates a background context with a timeout for async goroutines.
// The returned cancel function MUST be deferred by the goroutine that calls this.
func genCtx() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), 30*time.Second)
}
