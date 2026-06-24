package handler

import (
	"encoding/json"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/raiki02/video-extractor/internal/agent"
	"github.com/raiki02/video-extractor/internal/tool"
)

type ChatHandler struct {
	registry *tool.Registry
}

func NewChatHandler(registry *tool.Registry) *ChatHandler {
	return &ChatHandler{registry: registry}
}

type ChatRequest struct {
	Query     string `json:"query" binding:"required"`
	UserID    string `json:"user_id" binding:"required"`
	SessionID string `json:"session_id"`
}

type ChatResponse struct {
	TraceID string `json:"trace_id"`
	Answer  string `json:"answer"`
	Chunks  []any  `json:"chunks,omitempty"`
}

// ChatQuery handles POST /chat/query — RAG-based Q&A.
func (h *ChatHandler) ChatQuery(c *gin.Context) {
	var req ChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "query and user_id are required"})
		return
	}

	if req.SessionID == "" {
		req.SessionID = uuid.New().String()
	}

	traceID := uuid.New().String()

	result, err := agent.ExecuteChatQuery(c.Request.Context(), h.registry, req.Query, req.UserID, req.SessionID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error(), "trace_id": traceID})
		return
	}

	// Try to parse the RAG result
	var ragResult []any
	_ = json.Unmarshal([]byte(result), &ragResult)

	c.JSON(http.StatusOK, ChatResponse{
		TraceID: traceID,
		Answer:  result,
		Chunks:  ragResult,
	})
}
