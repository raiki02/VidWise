package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/raiki02/vidwise/internal/appconfig"
	"github.com/raiki02/vidwise/internal/chat"
	"github.com/raiki02/vidwise/internal/mcp"
	"github.com/raiki02/vidwise/internal/memory"
	"github.com/raiki02/vidwise/internal/model"
	"github.com/raiki02/vidwise/internal/rag"
	"github.com/raiki02/vidwise/internal/server"
	mysqlclient "github.com/raiki02/vidwise/internal/storage/mysql"
	qdrantclient "github.com/raiki02/vidwise/internal/storage/qdrant"
	"github.com/raiki02/vidwise/internal/tool"
)

const configPath = "config.yaml"

func main() {
	mode := flag.String("mode", "gateway", "run mode: gateway|worker")
	flag.Parse()

	cfg, err := appconfig.Load(configPath)
	if err != nil {
		panic(fmt.Errorf("load config %s failed: %w", configPath, err))
	}

	switch *mode {
	case "gateway":
		runGateway(cfg)
	case "worker":
		runWorker(cfg)
	default:
		panic(fmt.Errorf("unknown mode: %s (expected gateway or worker)", *mode))
	}
}

func runGateway(cfg appconfig.Config) {
	registry := tool.NewRegistry()

	// Connect to Qdrant
	var qdConn *qdrantclient.Client
	var embedClient *model.EmbedClient
	var rerankClient *model.RerankClient

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	qc, err := qdrantclient.NewClient(ctx, cfg.Qdrant.Addr())
	if err != nil {
		slog.Warn("gateway.qdrant_unavailable", "addr", cfg.Qdrant.Addr(), "err", err)
	} else {
		qdConn = qc
		slog.Info("gateway.qdrant_connected", "addr", cfg.Qdrant.Addr())
	}

	// Create embedding client
	ec, err := model.NewEmbedClient(cfg.Embedding)
	if err != nil {
		slog.Warn("gateway.embedding_unavailable", "err", err)
	} else {
		embedClient = ec
	}

	// Create rerank client
	rc, err := model.NewRerankClient(cfg.Rerank)
	if err != nil {
		slog.Warn("gateway.rerank_unavailable", "err", err)
	} else {
		rerankClient = rc
	}

	// Connect to MySQL and run migrations
	var chatRepo *chat.Repo
	var memRepo *memory.Repo
	if cfg.MySQL.DSN != "" {
		mysqlCtx, mysqlCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer mysqlCancel()
		mc, err := mysqlclient.NewClient(cfg.MySQL.DSN, cfg.MySQL.MaxOpen, cfg.MySQL.MaxIdle)
		if err != nil {
			slog.Warn("gateway.mysql_unavailable", "err", err)
		} else {
			chatRepo = chat.NewRepo(mc.DB)
			if err := chatRepo.AutoMigrate(); err != nil {
				slog.Warn("gateway.chat_migration_failed", "err", err)
			} else {
				slog.Info("gateway.mysql_ready")
			}

			memRepo = memory.NewRepo(mc.DB)
			if err := memRepo.AutoMigrate(); err != nil {
				slog.Warn("gateway.memory_migration_failed", "err", err)
			} else {
				slog.Info("gateway.memory_ready")
			}
			// Clean up mysql context
			_ = mysqlCtx
		}
	} else {
		slog.Warn("gateway.mysql_skipped", "reason", "no DSN configured")
	}

	// Ensure Qdrant collection exists with correct vector dimension
	// (detected from the actual embedding model at runtime).
	if qdConn != nil && embedClient != nil {
		indexer := rag.NewIndexer(embedClient, qdConn, cfg.Qdrant.Collection)
		if err := indexer.EnsureCollection(ctx); err != nil {
			slog.Warn("gateway.qdrant_ensure_collection_failed", "err", err)
		}
	}

	// Register tools
	registerTools(registry, cfg, qdConn, embedClient, rerankClient)

	// Start MCP server if enabled
	if cfg.MCP.Enabled {
		mcpSrv := mcp.New(cfg.MCP.Addr, cfg.MCP.Mode, registry)
		mcpSrv.StartAsync()
	}

	// Build and start Gin engine
	e := server.Router(cfg, registry, qdConn, embedClient, rerankClient, chatRepo, memRepo)
	slog.Info("gateway.starting", "addr", cfg.Server.Addr,
		"qdrant", qdConn != nil,
		"embedding", embedClient != nil,
		"rerank", rerankClient != nil,
		"mysql", chatRepo != nil,
		"memory", memRepo != nil,
		"mcp_enabled", cfg.MCP.Enabled,
	)

	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
		<-sig
		slog.Info("gateway.shutting_down")
		if qdConn != nil {
			_ = qdConn.Close()
		}
		os.Exit(0)
	}()

	if err := e.Run(cfg.Server.Addr); err != nil {
		panic(err)
	}
}

func registerTools(registry *tool.Registry, cfg appconfig.Config, qdConn *qdrantclient.Client, embedClient *model.EmbedClient, rerankClient *model.RerankClient) {
	// Download & audio extraction tools (no external dependencies)
	dlInner, dlWrap, err := tool.NewDownloadTool()
	if err != nil {
		slog.Warn("gateway.tool_register_failed", "tool", "download_video", "err", err)
	} else {
		registry.Register("download_video", dlInner, nil)
		_ = dlWrap
	}

	audioInner, audioWrap, err := tool.NewAudioExtractTool()
	if err != nil {
		slog.Warn("gateway.tool_register_failed", "tool", "extract_audio", "err", err)
	} else {
		registry.Register("extract_audio", audioInner, nil)
		_ = audioWrap
	}

	// Text format tool
	formatInner, formatWrap, err := tool.NewTextFormatTool(cfg.LLM)
	if err != nil {
		slog.Warn("gateway.tool_register_failed", "tool", "format_transcript", "err", err)
	} else {
		registry.Register("format_transcript", formatInner, nil)
		_ = formatWrap
	}

	// ASR tool is registered on-demand by the transcript agent during extraction flow.

	// RAG tools (require Qdrant + embedding)
	if qdConn != nil && embedClient != nil {
		// Embedding tool
		embInner, embWrap, err := tool.NewEmbedTool(embedClient)
		if err != nil {
			slog.Warn("gateway.tool_register_failed", "tool", "embed_texts", "err", err)
		} else {
			registry.Register("embed_texts", embInner, nil)
			_ = embWrap
		}

		// Rerank tool
		if rerankClient != nil {
			rerankInner, rerankWrap, err := tool.NewRerankTool(rerankClient)
			if err != nil {
				slog.Warn("gateway.tool_register_failed", "tool", "rerank_documents", "err", err)
			} else {
				registry.Register("rerank_documents", rerankInner, nil)
				_ = rerankWrap
			}
		}

		// RAG indexer
		indexer := rag.NewIndexer(embedClient, qdConn, cfg.Qdrant.Collection)
		if indexer != nil {
			ragIdxInner, ragIdxWrap, err := tool.NewRAGIndexTool(indexer)
			if err != nil {
				slog.Warn("gateway.tool_register_failed", "tool", "rag_index", "err", err)
			} else {
				registry.Register("rag_index", ragIdxInner, nil)
				_ = ragIdxWrap
			}
		}

		// RAG query tool
		retriever := rag.NewRetriever(embedClient, rerankClient, qdConn, cfg.Qdrant.Collection)
		ragQueryInner, ragQueryWrap, err := tool.NewRAGQueryTool(retriever)
		if err != nil {
			slog.Warn("gateway.tool_register_failed", "tool", "rag_query", "err", err)
		} else {
			registry.Register("rag_query", ragQueryInner, nil)
			_ = ragQueryWrap
		}
	}

	slog.Info("gateway.tools_registered", "count", len(registry.List()))
}

func runWorker(cfg appconfig.Config) {
	slog.Info("worker.starting")
	slog.Info("worker.waiting_for_tasks")

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig
	slog.Info("worker.shutting_down")
}
