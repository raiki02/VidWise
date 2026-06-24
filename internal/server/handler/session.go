package handler

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

type SessionHandler struct{}

func NewSessionHandler() *SessionHandler {
	return &SessionHandler{}
}

// GetSession handles GET /session/:id — returns session info.
func (h *SessionHandler) GetSession(c *gin.Context) {
	sessionID := c.Param("id")
	if sessionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "session id is required"})
		return
	}

	// In production: query MySQL session table
	c.JSON(http.StatusOK, gin.H{
		"session_id": sessionID,
		"status":     "active",
		"message":    "Session system is active. Connect MySQL for full session tracking.",
	})
}
