package handler

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/raiki02/vidwise/internal/appconfig"
	"github.com/raiki02/vidwise/internal/background"
	"github.com/raiki02/vidwise/internal/capability"
	"github.com/raiki02/vidwise/internal/chat"
	"github.com/raiki02/vidwise/internal/chatagent"
	"github.com/raiki02/vidwise/internal/memory"
	"github.com/raiki02/vidwise/internal/rag"
)

type ChatHandler struct {
	repo        *chat.Repo
	memRepo     *memory.Repo
	llmCfg      appconfig.LLMConfig
	caps        capability.Snapshot
	answerAgent *chatagent.Agent
	background  *background.Runner
}

func NewChatHandler(repo *chat.Repo, memRepo *memory.Repo, retriever *rag.Retriever, llmCfg appconfig.LLMConfig, caps capability.Snapshot) *ChatHandler {
	return NewChatHandlerWithRAGContext(repo, memRepo, retriever, llmCfg, caps, rag.DefaultContextConfig())
}

func NewChatHandlerWithRAGContext(repo *chat.Repo, memRepo *memory.Repo, retriever *rag.Retriever, llmCfg appconfig.LLMConfig, caps capability.Snapshot, ragContext rag.ContextConfig) *ChatHandler {
	return NewChatHandlerWithBackground(repo, memRepo, retriever, llmCfg, caps, ragContext, nil)
}

func NewChatHandlerWithBackground(repo *chat.Repo, memRepo *memory.Repo, retriever *rag.Retriever, llmCfg appconfig.LLMConfig, caps capability.Snapshot, ragContext rag.ContextConfig, runner *background.Runner) *ChatHandler {
	if runner == nil {
		runner = background.NewRunner(30 * time.Second)
	}
	return &ChatHandler{
		repo:    repo,
		memRepo: memRepo,
		llmCfg:  llmCfg,
		caps:    caps,
		answerAgent: chatagent.NewWithRetriever(
			llmCfg,
			ragContext,
			retriever,
		),
		background: runner,
	}
}

// ---- Request / Response types ----

type ChatQueryRequest struct {
	SessionID string `json:"session_id"`
	UserID    string `json:"user_id"` // optional, for cross-session memory
	Query     string `json:"query" binding:"required"`
}

type ChatChunk struct {
	SnippetNumber int     `json:"snippet_number,omitempty"`
	Text          string  `json:"text"`
	Score         float64 `json:"score"`
	SourceID      string  `json:"source_id,omitempty"`
	DocumentID    string  `json:"document_id,omitempty"`
	ChunkID       string  `json:"chunk_id,omitempty"`
	ContentHash   string  `json:"content_hash,omitempty"`
	TaskID        string  `json:"task_id,omitempty"`
	SessionID     string  `json:"session_id,omitempty"`
	ChunkIdx      int64   `json:"chunk_idx,omitempty"`
	SourceName    string  `json:"source_name,omitempty"`
	SourceURL     string  `json:"source_url,omitempty"`
	ContentType   string  `json:"content_type,omitempty"`
	DocumentTitle string  `json:"document_title,omitempty"`
	HeadingPath   string  `json:"heading_path,omitempty"`
}

type ChatQueryResponse struct {
	TraceID              string      `json:"trace_id,omitempty"`
	SessionID            string      `json:"session_id"`
	Answer               string      `json:"answer"`
	Chunks               []ChatChunk `json:"chunks,omitempty"`
	RAGTriggered         bool        `json:"rag_triggered"`
	RAGReason            string      `json:"rag_reason,omitempty"`
	RAGStatus            string      `json:"rag_status,omitempty"`
	RAGQuery             string      `json:"rag_query,omitempty"`
	RAGChunkCount        int         `json:"rag_chunk_count"`
	RAGContextUsedChunks int         `json:"rag_context_used_chunks"`
	RAGContextTruncated  bool        `json:"rag_context_truncated"`
	RAGContextChunks     []ChatChunk `json:"rag_context_chunks,omitempty"`
	Question             string      `json:"question"`
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

// ---- ChatQuery ----

func (h *ChatHandler) ChatQuery(c *gin.Context) {
	var req ChatQueryRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		errorJSON(c, http.StatusBadRequest, "query is required")
		return
	}

	ctx := c.Request.Context()
	req.Query = strings.TrimSpace(req.Query)
	if req.Query == "" {
		errorJSON(c, http.StatusBadRequest, "query is required")
		return
	}
	requestedSessionID := strings.TrimSpace(req.SessionID)
	sessionID := requestedSessionID

	// Auto-create session if none provided
	if sessionID == "" && h.sessionStoreAvailable() {
		title := "新对话"
		s, err := h.repo.CreateSessionForUser(ctx, req.UserID, title)
		if err != nil {
			slog.Error("chat.create_session_failed", "err", err)
			errorJSON(c, http.StatusInternalServerError, "创建会话失败")
			return
		}
		sessionID = s.ID
	}

	// Save user message
	if h.sessionStoreAvailable() && sessionID != "" {
		if _, err := h.repo.AddMessage(ctx, sessionID, "user", req.Query); err != nil {
			slog.Error("chat.save_user_msg_failed", "err", err)
		}
	}

	// Auto-title: generate on first user message
	if h.sessionStoreAvailable() && sessionID != "" {
		msgs, _ := h.repo.GetMessages(ctx, sessionID, 0)
		userCount := 0
		for _, m := range msgs {
			if m.Role == "user" {
				userCount++
			}
		}
		if userCount == 1 {
			h.background.Go("chat.auto_title", func(ctx context.Context) {
				h.autoGenerateTitle(ctx, sessionID, req.Query)
			})
		}
	}

	// Load user memory facts for cross-session context
	userFacts := ""
	if h.memoryStoreAvailable() && req.UserID != "" {
		userFacts = h.memRepo.FormatForPrompt(ctx, req.UserID)
	}

	// Build recent history for context
	var recentMsgs []chat.Message
	if h.sessionStoreAvailable() && sessionID != "" {
		recentMsgs, _ = h.repo.GetRecentMessages(ctx, sessionID, 20)
	}
	recentHistory := buildHistoryText(recentMsgs)

	turn, err := h.answerAgent.RunTurn(ctx, chatagent.TurnRequest{
		Query:     req.Query,
		History:   recentHistory,
		UserFacts: userFacts,
		UserID:    req.UserID,
		SessionID: sessionID,
	})
	if err != nil {
		errorJSON(c, http.StatusBadRequest, err.Error())
		return
	}
	chunks := turn.Chunks
	answer := turn.Answer

	// Save assistant response
	if h.sessionStoreAvailable() && sessionID != "" {
		if _, err := h.repo.AddMessage(ctx, sessionID, "assistant", answer); err != nil {
			slog.Error("chat.save_assistant_msg_failed", "err", err)
		}
	}

	// Step 4: Asynchronously extract/update user memory facts
	if h.memoryStoreAvailable() && req.UserID != "" {
		h.background.Go("chat.memory_extract", func(ctx context.Context) {
			h.extractMemoryFacts(ctx, req.UserID, sessionID, req.Query, answer, recentHistory)
		})
	}

	outChunks := make([]ChatChunk, 0)
	for _, c := range chunks {
		outChunks = append(outChunks, chatChunkFromRelevant(c))
	}
	contextChunks := make([]ChatChunk, 0, len(turn.RAGContext.Citations))
	for _, c := range turn.RAGContext.Citations {
		contextChunks = append(contextChunks, chatChunkFromContextCitation(c))
	}

	c.JSON(http.StatusOK, ChatQueryResponse{
		TraceID:              requestTraceID(c),
		SessionID:            sessionID,
		Answer:               answer,
		Chunks:               outChunks,
		RAGTriggered:         turn.Evaluation.ShouldRetrieve,
		RAGReason:            turn.Evaluation.Reason,
		RAGStatus:            string(turn.Retrieval.Status),
		RAGQuery:             turn.Retrieval.Query,
		RAGChunkCount:        turn.Retrieval.ChunkCount,
		RAGContextUsedChunks: turn.RAGContext.UsedChunks,
		RAGContextTruncated:  turn.RAGContext.Truncated,
		RAGContextChunks:     contextChunks,
		Question:             req.Query,
	})
}

func chatChunkFromRelevant(c rag.RelevantChunk) ChatChunk {
	return ChatChunk{
		Text:          c.Text,
		Score:         c.Score,
		SourceID:      c.SourceID,
		DocumentID:    c.DocumentID,
		ChunkID:       c.ChunkID,
		ContentHash:   c.ContentHash,
		TaskID:        c.TaskID,
		SessionID:     c.SessionID,
		ChunkIdx:      c.ChunkIdx,
		SourceName:    c.SourceName,
		SourceURL:     c.SourceURL,
		ContentType:   c.ContentType,
		DocumentTitle: c.DocumentTitle,
		HeadingPath:   c.HeadingPath,
	}
}

func chatChunkFromContextCitation(c rag.ContextCitation) ChatChunk {
	chunk := chatChunkFromRelevant(c.Chunk)
	chunk.SnippetNumber = c.SnippetNumber
	return chunk
}

// ---- Memory Extraction ----

// extractMemoryFacts runs asynchronously after each turn to extract
// or update facts about the user from the conversation.
func (h *ChatHandler) extractMemoryFacts(ctx context.Context, userID, sessionID, userMsg, assistantReply, recentHistory string) {
	if h.answerAgent == nil {
		return
	}

	existingJSON := "[]"
	if h.memRepo != nil {
		existingJSON = h.memRepo.DumpForPromptJSON(ctx, userID)
	}

	facts := h.answerAgent.ExtractMemoryFacts(ctx, chatagent.MemoryExtractionRequest{
		ExistingFactsJSON: existingJSON,
		History:           recentHistory,
		UserMessage:       userMsg,
		AssistantReply:    assistantReply,
	})
	if len(facts) == 0 {
		slog.Info("memory.no_facts_extracted", "user_id", userID)
		return
	}

	result, err := h.memRepo.ApplyExtractedFacts(ctx, userID, sessionID, facts)
	if err != nil {
		slog.Error("memory.apply_failed", "err", err)
		return
	}

	slog.Info("memory.extraction_done", "user_id", userID, "facts", len(facts), "applied", result.Applied, "skipped", result.Skipped)
}

// ---- Sessions ----

func (h *ChatHandler) ListSessions(c *gin.Context) {
	if !h.hasSessionStore(c) {
		return
	}

	userID := c.Query("user_id")

	var sessions []chat.Session
	var err error

	if userID != "" {
		sessions, err = h.repo.ListSessionsByUser(c.Request.Context(), userID, 50)
	} else {
		sessions, err = h.repo.ListSessions(c.Request.Context(), 50)
	}

	if err != nil {
		errorJSON(c, http.StatusInternalServerError, "获取会话列表失败")
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
	if !h.hasSessionStore(c) {
		return
	}

	sessionID := c.Param("id")
	if sessionID == "" {
		errorJSON(c, http.StatusBadRequest, "session id is required")
		return
	}

	session, err := h.repo.GetSession(c.Request.Context(), sessionID)
	if err != nil {
		errorJSON(c, http.StatusNotFound, "会话不存在")
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
	if !h.hasSessionStore(c) {
		return
	}

	var req struct {
		UserID string `json:"user_id"`
	}
	_ = c.ShouldBindJSON(&req)

	s, err := h.repo.CreateSessionForUser(c.Request.Context(), req.UserID, "新对话")
	if err != nil {
		errorJSON(c, http.StatusInternalServerError, "创建会话失败")
		return
	}
	c.JSON(http.StatusCreated, gin.H{
		"session_id": s.ID,
		"user_id":    s.UserID,
		"title":      s.Title,
	})
}

func (h *ChatHandler) RAGHealth(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"rag":          h.caps.LegacyStatus(capability.RAG),
		"llm":          h.llmCfg.Enabled,
		"llm_status":   h.caps.Get(capability.LLM).Status,
		"memory":       h.caps.Get(capability.MemoryStore).Status,
		"chat":         h.caps.Get(capability.ChatSessionStore).Status,
		"capabilities": h.caps.Map(),
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
		errorJSON(c, http.StatusBadRequest, "user_id is required")
		return
	}
	if !h.memoryStoreAvailable() {
		errorJSONWithFields(c, http.StatusServiceUnavailable, "memory service unavailable", gin.H{"capability": h.caps.Get(capability.MemoryStore)})
		return
	}

	facts, err := h.memRepo.GetFactsByUser(c.Request.Context(), userID)
	if err != nil {
		errorJSON(c, http.StatusInternalServerError, "获取用户画像失败")
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

func (h *ChatHandler) hasSessionStore(c *gin.Context) bool {
	if h.sessionStoreAvailable() {
		return true
	}
	capabilityStatus := h.caps.Get(capability.ChatSessionStore)
	errorJSONWithFields(c, http.StatusServiceUnavailable, "chat session store unavailable", gin.H{"capability": capabilityStatus})
	return false
}

func (h *ChatHandler) sessionStoreAvailable() bool {
	return h.repo != nil && h.caps.Available(capability.ChatSessionStore)
}

func (h *ChatHandler) memoryStoreAvailable() bool {
	return h.memRepo != nil && h.caps.Available(capability.MemoryStore)
}

func (h *ChatHandler) autoGenerateTitle(ctx context.Context, sessionID, firstQuery string) {
	if !h.sessionStoreAvailable() {
		return
	}
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
