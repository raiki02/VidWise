package server

import (
	"embed"
	"io/fs"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/raiki02/vidwise/internal/chat"
)

//go:embed web/index.html
var webFS embed.FS

// Router builds the MVP API: health, agent chat, and the single-page UI.
func Router(chatService *chat.Service) *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	e := gin.New()
	e.Use(gin.Recovery())

	e.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})

	h := &ChatHandler{service: chatService}
	e.POST("/agent/chat", h.AgentChat)

	index, err := fs.ReadFile(webFS, "web/index.html")
	if err == nil {
		e.GET("/", func(c *gin.Context) {
			c.Data(http.StatusOK, "text/html; charset=utf-8", index)
		})
	}
	return e
}
