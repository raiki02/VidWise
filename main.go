package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	einotool "github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/flow/agent/react"
	"github.com/cloudwego/eino/schema"
	"github.com/raiki02/vidwise/internal/agent"
	"github.com/raiki02/vidwise/internal/appconfig"
	"github.com/raiki02/vidwise/internal/asr"
	"github.com/raiki02/vidwise/internal/chat"
	"github.com/raiki02/vidwise/internal/embedding"
	"github.com/raiki02/vidwise/internal/llm"
	"github.com/raiki02/vidwise/internal/server"
	mysqlclient "github.com/raiki02/vidwise/internal/storage/mysql"
	apptool "github.com/raiki02/vidwise/internal/tool"
	"github.com/raiki02/vidwise/internal/vectorstore"
)

const configPath = "config.yaml"

// agentAdapter adapts the eino react agent (Generate has a variadic options
// parameter) to the chat.Generator interface.
type agentAdapter struct {
	agent *react.Agent
}

func (a agentAdapter) Generate(ctx context.Context, messages []*schema.Message) (*schema.Message, error) {
	return a.agent.Generate(ctx, messages)
}

func main() {
	cfg, err := appconfig.Load(configPath)
	if err != nil {
		panic(fmt.Errorf("load config %s failed: %w", configPath, err))
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	// 1. LLM is mandatory: the agent cannot run without it.
	chatModel, err := llm.NewChatModel(ctx, cfg.LLM)
	if err != nil {
		panic(fmt.Errorf("create llm chat model: %w", err))
	}

	// 2. ASR is mandatory for the closed loop.
	asrClient, err := asr.NewClient(cfg.ASR.BaseURL, cfg.ASR.Language, cfg.ASR.TimeoutDurationOr(10*time.Minute), asr.TranscribeOptions{
		BeamSize:      cfg.ASR.Transcribe.BeamSize,
		InitialPrompt: cfg.ASR.Transcribe.InitialPrompt,
	})
	if err != nil {
		panic(fmt.Errorf("create asr client: %w", err))
	}
	if err := asrClient.Health(ctx); err != nil {
		panic(fmt.Errorf("asr service unavailable at %s: %w", cfg.ASR.BaseURL, err))
	}
	slog.Info("asr.ready", "base_url", cfg.ASR.BaseURL)

	// 3. Embedding service is mandatory for the RAG tools.
	embedClient, err := embedding.NewClient(cfg.Embedding.BaseURL, cfg.Embedding.Model, cfg.Embedding.TimeoutDurationOr(2*time.Minute))
	if err != nil {
		panic(fmt.Errorf("create embedding client: %w", err))
	}
	if err := embedClient.Health(ctx); err != nil {
		panic(fmt.Errorf("embedding service unavailable at %s: %w", cfg.Embedding.BaseURL, err))
	}
	slog.Info("embedding.ready", "base_url", cfg.Embedding.BaseURL)

	// 4. Qdrant is mandatory for the RAG tools.
	store, err := vectorstore.New(ctx, cfg.Qdrant.Addr(), cfg.Qdrant.Collection, uint64(cfg.Qdrant.VectorDim))
	if err != nil {
		panic(fmt.Errorf("qdrant unavailable at %s: %w", cfg.Qdrant.Addr(), err))
	}
	defer store.Close()

	// 5. Chat history store: MySQL if configured, in-memory otherwise.
	var history chat.HistoryStore
	var mysqlClient *mysqlclient.Client
	if cfg.MySQL.DSN != "" {
		mysqlClient, err = mysqlclient.NewClient(cfg.MySQL.DSN, cfg.MySQL.MaxOpen, cfg.MySQL.MaxIdle)
		if err != nil {
			panic(fmt.Errorf("mysql unavailable: %w", err))
		}
		defer mysqlClient.Close()
		repo := chat.NewRepo(mysqlClient.DB)
		if err := repo.AutoMigrate(); err != nil {
			panic(fmt.Errorf("chat migration failed: %w", err))
		}
		history = repo
		slog.Info("mysql.ready")
	} else {
		history = chat.NewMemoryStore()
		slog.Warn("mysql.skipped_dsn_empty", "info", "using in-memory chat history")
	}

	// 6. Build tools.
	workDir := cfg.Download.WorkDir
	if err := os.MkdirAll(workDir, 0o755); err != nil {
		panic(fmt.Errorf("create work dir: %w", err))
	}
	downloadTool, err := apptool.NewDownloadAudioTool(workDir, cfg.Download.CookiesPath)
	if err != nil {
		panic(fmt.Errorf("build download_audio tool: %w", err))
	}
	transcribeTool, err := apptool.NewTranscribeTool(workDir, asrClient)
	if err != nil {
		panic(fmt.Errorf("build transcribe_audio tool: %w", err))
	}
	indexTool, err := apptool.NewRAGIndexTool(workDir, cfg.RAG.ChunkRunes, cfg.RAG.OverlapRunes, embedClient, store)
	if err != nil {
		panic(fmt.Errorf("build index_transcript tool: %w", err))
	}
	queryTool, err := apptool.NewRAGQueryTool(embedClient, store, cfg.RAG.TopK)
	if err != nil {
		panic(fmt.Errorf("build search_knowledge tool: %w", err))
	}

	// 7. Build the agent and the chat service.
	chatAgent, err := agent.New(ctx, chatModel, []einotool.BaseTool{downloadTool, transcribeTool, indexTool, queryTool}, cfg.Agent.MaxStep)
	if err != nil {
		panic(fmt.Errorf("build agent: %w", err))
	}
	service := chat.NewService(history, agentAdapter{agent: chatAgent}, cfg.Agent.HistoryLimit)

	// 8. HTTP server.
	e := server.Router(service)
	httpServer, err := server.NewHTTPServer(cfg.Server, e)
	if err != nil {
		panic(fmt.Errorf("build http server: %w", err))
	}
	slog.Info("gateway.starting", "addr", cfg.Server.Addr, "llm_provider", cfg.LLM.Provider, "asr", cfg.ASR.BaseURL, "embedding", cfg.Embedding.BaseURL, "qdrant", cfg.Qdrant.Addr(), "mysql", mysqlClient != nil)

	errCh := make(chan error, 1)
	go func() {
		errCh <- httpServer.ListenAndServe()
	}()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(sigCh)

	select {
	case err := <-errCh:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			panic(err)
		}
	case sig := <-sigCh:
		slog.Info("gateway.shutting_down", "signal", sig.String(), "timeout", server.ShutdownTimeout(cfg.Server))
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), server.ShutdownTimeout(cfg.Server))
		defer shutdownCancel()
		if err := httpServer.Shutdown(shutdownCtx); err != nil {
			slog.Error("gateway.shutdown_failed", "err", err)
			_ = httpServer.Close()
		}
		if err := <-errCh; err != nil && !errors.Is(err, http.ErrServerClosed) {
			panic(err)
		}
	}
}
