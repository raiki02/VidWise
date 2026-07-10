package server

import (
	"embed"
	"io/fs"
	"log/slog"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/raiki02/vidwise/internal/appconfig"
	"github.com/raiki02/vidwise/internal/background"
	"github.com/raiki02/vidwise/internal/capability"
	"github.com/raiki02/vidwise/internal/chat"
	"github.com/raiki02/vidwise/internal/memory"
	"github.com/raiki02/vidwise/internal/ragruntime"
	"github.com/raiki02/vidwise/internal/server/handler"
	"github.com/raiki02/vidwise/internal/tool"

	_ "embed"
)

//go:embed web/*
var webFS embed.FS

// Router assembles all HTTP routes for the gateway.
// Pass nil for optional dependencies if not available.
func Router(cfg appconfig.Config, registry *tool.Registry, ragRuntime ragruntime.Runtime, chatRepo *chat.Repo, memRepo *memory.Repo, caps capability.Snapshot) *gin.Engine {
	e := gin.Default()

	e.Use(TraceID())
	e.Use(RequestLogger())

	// Static files
	web, err := fs.Sub(webFS, "web")
	if err != nil {
		panic(err)
	}
	e.StaticFS("/static", http.FS(web))

	// Health
	e.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":       "ok",
			"capabilities": caps.Map(),
		})
	})

	backgroundRunner := background.NewRunner(30 * time.Second)

	// Handlers
	extractHandler := handler.NewExtractHandlerWithSourceManagerAndBackground(cfg, registry, ragRuntime.Sources, caps, backgroundRunner)
	videoHandler := handler.NewVideoHandler(registry)
	taskHandler := handler.NewTaskHandler()

	if ragRuntime.Usable() {
		slog.Info("gateway.rag_ready",
			"search_top_k", ragRuntime.Retrieval.SearchTopK,
			"top_k", ragRuntime.Retrieval.TopK,
			"min_score", ragRuntime.Retrieval.MinScore,
		)
	} else {
		ragCapability := caps.Get(capability.RAG)
		slog.Warn("gateway.rag_unavailable", "status", ragCapability.Status, "reason", ragCapability.Reason)
	}

	chatHandler := handler.NewChatHandlerWithBackground(chatRepo, memRepo, ragRuntime.Retriever, cfg.LLM, caps, ragRuntime.Context, backgroundRunner)

	// Legacy endpoints (backward compatible)
	e.GET("/extract", extractHandler.Extract)
	e.POST("/extract", extractHandler.Extract)
	e.POST("/format", extractHandler.FormatText)

	// Video process
	e.POST("/video/process", videoHandler.VideoProcess)

	// Chat / sessions (session-based, no user_id)
	e.POST("/chat/new", chatHandler.NewSession)
	e.GET("/chat/sessions", chatHandler.ListSessions)
	e.GET("/chat/session/:id", chatHandler.GetSession)
	e.POST("/chat/query", chatHandler.ChatQuery)

	// Task status
	e.GET("/task/:id", taskHandler.GetTask)

	// Health
	e.GET("/rag/health", chatHandler.RAGHealth)

	// User memory / cross-session profile
	e.GET("/user/facts", chatHandler.GetUserFacts)
	e.GET("/user/profile", chatHandler.GetUserFacts) // alias

	// File upload — index text into knowledge base
	e.POST("/upload", extractHandler.UploadText)
	e.GET("/rag/sources", extractHandler.ListRAGSources)
	e.DELETE("/rag/source/:source_id", extractHandler.DeleteRAGSource)

	// Serve the chat UI at root
	e.GET("/", func(c *gin.Context) {
		c.Redirect(http.StatusMovedPermanently, "/static/index.html")
	})

	return e
}
