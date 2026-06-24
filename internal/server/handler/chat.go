package handler

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"strings"

	"github.com/cloudwego/eino/schema"
	"github.com/gin-gonic/gin"
	"github.com/raiki02/vidwise/internal/appconfig"
	"github.com/raiki02/vidwise/internal/chat"
	"github.com/raiki02/vidwise/internal/paragraph"
	"github.com/raiki02/vidwise/internal/rag"
)

type ChatHandler struct {
	repo      *chat.Repo
	retriever *rag.Retriever
	llmCfg    appconfig.LLMConfig
}

func NewChatHandler(repo *chat.Repo, retriever *rag.Retriever, llmCfg appconfig.LLMConfig) *ChatHandler {
	return &ChatHandler{repo: repo, retriever: retriever, llmCfg: llmCfg}
}

// ---- Request / Response types ----

type ChatQueryRequest struct {
	SessionID string `json:"session_id"`
	Query     string `json:"query" binding:"required"`
}

type ChatChunk struct {
	Text  string  `json:"text"`
	Score float64 `json:"score"`
}

type ChatQueryResponse struct {
	SessionID string      `json:"session_id"`
	Answer    string      `json:"answer"`
	Chunks    []ChatChunk `json:"chunks,omitempty"`
	Question  string      `json:"question"`
}

type SessionListItem struct {
	ID        string `json:"id"`
	Title     string `json:"title"`
	UpdatedAt string `json:"updated_at"`
}

type SessionDetail struct {
	ID       string         `json:"id"`
	Title    string         `json:"title"`
	Messages []chat.Message `json:"messages"`
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
		s, err := h.repo.CreateSession(ctx, "新对话")
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

	// Auto-title: if this is the first message, generate a title
	msgs, _ := h.repo.GetMessages(ctx, sessionID, 0)
	userCount := 0
	for _, m := range msgs {
		if m.Role == "user" {
			userCount++
		}
	}
	if userCount == 1 {
		go h.autoGenerateTitle(context.Background(), sessionID, req.Query)
	}

	// Step 1: RAG retrieval
	chunks, err := h.retriever.Retrieve(ctx, req.Query)
	if err != nil {
		slog.Error("chat.retrieve_failed", "err", err)
	}

	// Step 2: Build context from RAG + chat history
	answer := h.buildAnswer(ctx, sessionID, chunks, req.Query)

	// Save assistant response
	if _, err := h.repo.AddMessage(ctx, sessionID, "assistant", answer); err != nil {
		slog.Error("chat.save_assistant_msg_failed", "err", err)
	}

	outChunks := make([]ChatChunk, 0)
	for _, c := range chunks {
		outChunks = append(outChunks, ChatChunk{Text: c.Text, Score: c.Score})
	}

	c.JSON(http.StatusOK, ChatQueryResponse{
		SessionID: sessionID,
		Answer:    answer,
		Chunks:    outChunks,
		Question:  req.Query,
	})
}

func (h *ChatHandler) buildAnswer(ctx context.Context, sessionID string, chunks []rag.RelevantChunk, query string) string {
	// Build RAG context
	ragContext := ""
	for i, c := range chunks {
		if i > 0 {
			ragContext += "\n---\n"
		}
		ragContext += c.Text
	}

	// Fetch recent chat history
	var history []chat.Message
	if h.repo != nil && sessionID != "" {
		history, _ = h.repo.GetRecentMessages(ctx, sessionID, 30)
	}

	// Build conversation history text
	historyText := ""
	for _, m := range history {
		if m.Role == "user" {
			historyText += fmt.Sprintf("用户: %s\n", m.Content)
		} else {
			historyText += fmt.Sprintf("助手: %s\n", m.Content)
		}
	}

	// Try LLM
	if h.llmCfg.Enabled != nil && *h.llmCfg.Enabled && h.llmCfg.Model != "" {
		cm, err := paragraph.NewChatModel(ctx, h.llmCfg)
		if err == nil {
			var msgs []*schema.Message
			msgs = append(msgs, schema.SystemMessage(`你是视频知识库问答助手。你的任务是：根据提供的视频转录文本上下文和对话历史，准确、完整地回答用户的问题。

要求：
1. 仅根据提供的上下文回答，不要编造信息。
2. 提取上下文中与问题最相关的内容，用自己的话重新组织。
3. 回答要具体、有细节，不要泛泛而谈。
4. 如果上下文信息不足，明确告知用户，不要猜测。
5. 结合对话历史理解用户的意图和上下文。
6. 用中文回答，语言清晰易懂。`))

			userPrompt := fmt.Sprintf("视频转录文本：\n%s\n\n用户问题：%s", ragContext, query)
			if historyText != "" {
				userPrompt = fmt.Sprintf("对话历史：\n%s\n\n视频转录文本：\n%s\n\n用户最新问题：%s", historyText, ragContext, query)
			}
			userPrompt += "\n\n请根据视频内容和对话历史回答用户的最新问题。"
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

	// Fallback: return relevant context as-is
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

// ---- Sessions ----

func (h *ChatHandler) ListSessions(c *gin.Context) {
	sessions, err := h.repo.ListSessions(c.Request.Context(), 50)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取会话列表失败"})
		return
	}
	list := make([]SessionListItem, 0, len(sessions))
	for _, s := range sessions {
		list = append(list, SessionListItem{
			ID:        s.ID,
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
		Title:    session.Title,
		Messages: messages,
	})
}

func (h *ChatHandler) NewSession(c *gin.Context) {
	s, err := h.repo.CreateSession(c.Request.Context(), "新对话")
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "创建会话失败"})
		return
	}
	c.JSON(http.StatusCreated, gin.H{
		"session_id": s.ID,
		"title":      s.Title,
	})
}

func (h *ChatHandler) RAGHealth(c *gin.Context) {
	status := "available"
	if h.retriever == nil {
		status = "unavailable"
	}
	c.JSON(http.StatusOK, gin.H{"rag": status, "llm": h.llmCfg.Enabled})
}
