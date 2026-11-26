package server

import (
	"errors"
	"log/slog"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/raiki02/vidwise/internal/chat"
)

type ChatHandler struct {
	service *chat.Service
}

type agentChatRequest struct {
	SessionID string `json:"session_id"`
	Message   string `json:"message"`
}

type agentChatResponse struct {
	SessionID string `json:"session_id"`
	Reply     string `json:"reply"`
}

func (h *ChatHandler) AgentChat(c *gin.Context) {
	var req agentChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid JSON body"})
		return
	}
	if strings.TrimSpace(req.Message) == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "message is required"})
		return
	}

	sessionID, reply, err := h.service.Chat(c.Request.Context(), strings.TrimSpace(req.SessionID), req.Message)
	if err != nil {
		if errors.Is(err, chat.ErrSessionNotFound) {
			c.JSON(http.StatusNotFound, gin.H{"error": "session not found"})
			return
		}
		slog.Error("agent.chat_failed", "err", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "agent failed, please retry"})
		return
	}
	c.JSON(http.StatusOK, agentChatResponse{SessionID: sessionID, Reply: reply})
}
