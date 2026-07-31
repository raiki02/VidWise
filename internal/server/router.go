package server

import (
	"embed"
	"io/fs"
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
	"github.com/raiki02/vidwise/internal/knowledgeagent"
	"github.com/raiki02/vidwise/internal/memory"
	"github.com/raiki02/vidwise/internal/ragruntime"
	"github.com/raiki02/vidwise/internal/server/handler"
	taskpkg "github.com/raiki02/vidwise/internal/task"
	"github.com/raiki02/vidwise/internal/tool"

	_ "embed"
)

//go:embed web/*
var webFS embed.FS

// Router assembles all HTTP routes for the gateway.
// Pass nil for optional dependencies if not available.
func Router(cfg appconfig.Config, registry *tool.Registry, ragRuntime ragruntime.Runtime, chatRepo *chat.Repo, memRepo *memory.Repo, caps capability.Snapshot) *gin.Engine {
	return RouterWithTaskRepo(cfg, registry, ragRuntime, chatRepo, memRepo, nil, caps)
}

// RouterWithTaskRepo assembles routes with an optional MySQL-backed task store.
func RouterWithTaskRepo(cfg appconfig.Config, registry *tool.Registry, ragRuntime ragruntime.Runtime, chatRepo *chat.Repo, memRepo *memory.Repo, taskRepo *taskpkg.Repo, caps capability.Snapshot) *gin.Engine {
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
	e.GET("/ready", func(c *gin.Context) {
		readiness := caps.Readiness(capability.RAG, capability.LLM)
		status := http.StatusOK
		if !readiness.Ready {
			status = http.StatusServiceUnavailable
		}
		c.JSON(status, readiness)
	})
	e.GET("/api/capabilities", func(c *gin.Context) {
		c.JSON(http.StatusOK, BuildFrontendManifest(registry, caps))
	})

	backgroundRunner := background.NewRunner(30 * time.Second)
	taskTracker := newTaskTrackerFromConfigAndRepo(cfg, taskRepo)
	taskTrackerOptions := taskTrackerOptionsFromConfig(cfg)

	// Handlers
	extractHandler := handler.NewExtractHandlerWithSourceManagerAndBackground(cfg, registry, ragRuntime.Sources, caps, backgroundRunner)
	videoRunner := handler.NewVideoProcessRunner(cfg.Task.MaxConcurrentVideo)
	videoHandler := handler.NewVideoHandlerWithBackgroundAndTasks(registry, videoRunner, taskTracker)
	taskHandler := handler.NewTaskHandlerWithTracker(taskTracker)

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
	agentActionStore := knowledgeagent.NewActionStore(knowledgeagent.ActionStoreOptions{
		MaxActions: taskTrackerOptions.MaxTasks,
		RetainFor:  taskTrackerOptions.RetainFor,
	})
	var agentSessions knowledgeagent.SessionStore
	if chatRepo != nil {
		agentSessions = chatRepo
	}
	knowledgeAgent := knowledgeagent.NewService(knowledgeagent.ServiceConfig{
		Sessions:         agentSessions,
		Answerer:         chatagent.NewWithRetriever(cfg.LLM, ragRuntime.Context, ragRuntime.Retriever),
		Sources:          ragRuntime.Sources,
		Videos:           handler.KnowledgeVideoAdapter(videoHandler),
		Indexer:          handler.KnowledgeTranscriptIndexer(videoHandler),
		Formatter:        handler.KnowledgeTextFormatter(cfg.LLM),
		Tasks:            handler.KnowledgeTaskReader(taskTracker),
		Actions:          agentActionStore,
		IntentClassifier: knowledgeagent.NewLLMIntentClassifier(cfg.LLM),
	})
	agentHandler := handler.NewKnowledgeAgentHandler(knowledgeAgent)

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

	// VidWise video knowledge agent
	e.POST("/agent/turn", agentHandler.Turn)
	e.POST("/agent/actions/:id/confirm", agentHandler.ConfirmAction)

	// Task status
	e.GET("/tasks", taskHandler.ListTasks)
	e.GET("/task/:id", taskHandler.GetTask)
	e.POST("/task/:id/index", videoHandler.IndexTaskTranscript)

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

func newTaskTrackerFromConfig(cfg appconfig.Config) *taskpkg.Tracker {
	return newTaskTrackerFromConfigAndRepo(cfg, nil)
}

func newTaskTrackerFromConfigAndRepo(cfg appconfig.Config, repo *taskpkg.Repo) *taskpkg.Tracker {
	opts := taskTrackerOptionsFromConfig(cfg)
	if repo != nil {
		opts.Store = repo
		opts.StoragePath = ""
	}
	return taskpkg.NewTrackerWithOptions(opts)
}

func taskTrackerOptionsFromConfig(cfg appconfig.Config) taskpkg.TrackerOptions {
	retainFor, err := cfg.Task.RetentionDuration()
	if err != nil {
		slog.Warn("gateway.task_tracker_config_invalid", "retain_for", cfg.Task.RetainFor, "err", err)
		return taskpkg.TrackerOptions{
			MaxTasks:    cfg.Task.MaxTracked,
			StoragePath: strings.TrimSpace(cfg.Task.StoragePath),
		}
	}
	return taskpkg.TrackerOptions{
		MaxTasks:    cfg.Task.MaxTracked,
		RetainFor:   retainFor,
		StoragePath: strings.TrimSpace(cfg.Task.StoragePath),
	}
}
