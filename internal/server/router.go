package server

import (
	"embed"
	"io/fs"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/raiki02/video-extractor/internal/appconfig"
	"github.com/raiki02/video-extractor/internal/server/handler"
	"github.com/raiki02/video-extractor/internal/tool"

	// embed the web frontend
	_ "embed"
)

//go:embed web/*
var webFS embed.FS

// Router assembles all HTTP routes for the gateway.
func Router(cfg appconfig.Config, registry *tool.Registry) *gin.Engine {
	e := gin.Default()

	// Middleware
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
	extractHandler := handler.NewExtractHandler(cfg)
	videoHandler := handler.NewVideoHandler(registry)
	chatHandler := handler.NewChatHandler(registry)
	taskHandler := handler.NewTaskHandler()
	sessionHandler := handler.NewSessionHandler()

	// Legacy endpoints (backward compatible)
	e.GET("/extract", extractHandler.Extract)
	e.POST("/extract", extractHandler.Extract)
	e.POST("/format", extractHandler.FormatText)

	// New endpoints
	e.POST("/video/process", videoHandler.VideoProcess)
	e.POST("/chat/query", chatHandler.ChatQuery)
	e.GET("/task/:id", taskHandler.GetTask)
	e.GET("/session/:id", sessionHandler.GetSession)

	return e
}
