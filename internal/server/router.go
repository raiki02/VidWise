package server

import (
	"embed"
	"io/fs"
	"log/slog"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/raiki02/video-extractor/internal/appconfig"
	"github.com/raiki02/video-extractor/internal/chat"
	"github.com/raiki02/video-extractor/internal/model"
	"github.com/raiki02/video-extractor/internal/rag"
	"github.com/raiki02/video-extractor/internal/server/handler"
	qdrantclient "github.com/raiki02/video-extractor/internal/storage/qdrant"
	"github.com/raiki02/video-extractor/internal/tool"

	_ "embed"
)

//go:embed web/*
var webFS embed.FS

// Router assembles all HTTP routes for the gateway.
// Pass nil for optional dependencies if not available.
func Router(cfg appconfig.Config, registry *tool.Registry, qdConn *qdrantclient.Client, embedClient *model.EmbedClient, rerankClient *model.RerankClient, chatRepo *chat.Repo) *gin.Engine {
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
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})

	// Handlers
	extractHandler := handler.NewExtractHandler(cfg, registry, nil)
	videoHandler := handler.NewVideoHandler(registry)
	taskHandler := handler.NewTaskHandler()

	// Build RAG indexer and retriever if Qdrant and embedding are available
	var indexer *rag.Indexer
	var retriever *rag.Retriever
	if qdConn != nil && embedClient != nil {
		indexer = rag.NewIndexer(embedClient, qdConn, cfg.Qdrant.Collection)
		extractHandler = handler.NewExtractHandler(cfg, registry, indexer)
		retriever = rag.NewRetriever(embedClient, rerankClient, qdConn, cfg.Qdrant.Collection)
		slog.Info("gateway.rag_ready")
	} else {
		slog.Warn("gateway.rag_unavailable", "qdrant", qdConn != nil, "embedding", embedClient != nil)
	}

	chatHandler := handler.NewChatHandler(chatRepo, retriever, cfg.LLM)

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

	// Serve the chat UI at root
	e.GET("/", func(c *gin.Context) {
		c.Redirect(http.StatusMovedPermanently, "/static/index.html")
	})

	return e
}
