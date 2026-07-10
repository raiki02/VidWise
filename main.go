package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/raiki02/vidwise/internal/appconfig"
	"github.com/raiki02/vidwise/internal/asr"
	"github.com/raiki02/vidwise/internal/capability"
	"github.com/raiki02/vidwise/internal/chat"
	"github.com/raiki02/vidwise/internal/mcp"
	"github.com/raiki02/vidwise/internal/memory"
	"github.com/raiki02/vidwise/internal/model"
	"github.com/raiki02/vidwise/internal/ragregistry"
	"github.com/raiki02/vidwise/internal/ragruntime"
	"github.com/raiki02/vidwise/internal/server"
	mysqlclient "github.com/raiki02/vidwise/internal/storage/mysql"
	qdrantclient "github.com/raiki02/vidwise/internal/storage/qdrant"
	"github.com/raiki02/vidwise/internal/tool"
	video_summary "github.com/raiki02/vidwise/internal/video_summary"
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
	var asrClient *asr.Client
	asrReady := false
	videoSummaryReady := false

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
	} else if err := ec.Health(ctx); err != nil {
		slog.Warn("gateway.embedding_unhealthy", "base_url", cfg.Embedding.BaseURL, "err", err)
	} else {
		embedClient = ec
	}

	// Create rerank client
	rc, err := model.NewRerankClient(cfg.Rerank)
	if err != nil {
		slog.Warn("gateway.rerank_unavailable", "err", err)
	} else if err := rc.Health(ctx); err != nil {
		slog.Warn("gateway.rerank_unhealthy", "base_url", cfg.Rerank.BaseURL, "err", err)
	} else {
		rerankClient = rc
	}

	asrTimeout, err := cfg.ASR.TimeoutDuration()
	if err != nil {
		slog.Warn("gateway.asr_unavailable", "err", err)
	} else {
		client, err := asr.NewClient(cfg.ASR.BaseURL, cfg.ASR.Language, asrTimeout, asr.TranscribeOptions{
			BeamSize:      cfg.ASR.Transcribe.BeamSize,
			VADFilter:     cfg.ASR.Transcribe.VADFilter,
			InitialPrompt: cfg.ASR.Transcribe.InitialPrompt,
		})
		if err != nil {
			slog.Warn("gateway.asr_unavailable", "err", err)
		} else if err := client.Health(ctx); err != nil {
			asrClient = client
			slog.Warn("gateway.asr_unhealthy", "base_url", cfg.ASR.BaseURL, "err", err)
		} else {
			asrClient = client
			asrReady = true
		}
	}

	videoSummaryTimeout, err := cfg.VideoSummary.TimeoutDuration()
	if err != nil {
		slog.Warn("gateway.video_summary_unavailable", "err", err)
	} else {
		client, err := video_summary.NewClient(cfg.VideoSummary.BaseURL, videoSummaryTimeout, video_summary.SummarizeOptions{
			MaxNewTokens: cfg.VideoSummary.Summarize.MaxNewTokens,
			Prompt:       cfg.VideoSummary.Summarize.Prompt,
			DoSample:     cfg.VideoSummary.Summarize.DoSample,
			Temperature:  cfg.VideoSummary.Summarize.Temperature,
			TopP:         cfg.VideoSummary.Summarize.TopP,
		})
		if err != nil {
			slog.Warn("gateway.video_summary_unavailable", "err", err)
		} else if err := client.Health(ctx); err != nil {
			slog.Warn("gateway.video_summary_unhealthy", "base_url", cfg.VideoSummary.BaseURL, "err", err)
		} else {
			videoSummaryReady = true
		}
	}

	// Connect to MySQL and run migrations
	var chatRepo *chat.Repo
	var memRepo *memory.Repo
	var sourceRegistry *ragregistry.Repo
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

			sourceRegistry = ragregistry.NewRepo(mc.DB)
			if err := sourceRegistry.AutoMigrate(); err != nil {
				slog.Warn("gateway.rag_source_registry_migration_failed", "err", err)
				sourceRegistry = nil
			} else {
				slog.Info("gateway.rag_source_registry_ready")
			}
			// Clean up mysql context
			_ = mysqlCtx
		}
	} else {
		slog.Warn("gateway.mysql_skipped", "reason", "no DSN configured")
	}

	ragBuild := ragruntime.Build(ctx, cfg, ragruntime.Deps{
		Qdrant:   qdConn,
		Embed:    embedClient,
		Rerank:   rerankClient,
		Registry: sourceRegistry,
	})
	if ragBuild.Err != nil {
		slog.Warn("gateway.qdrant_ensure_collection_failed", "err", ragBuild.Err)
	}

	caps := capability.FromRuntime(capability.RuntimeDeps{
		ChatSessionStore: chatRepo != nil,
		MemoryStore:      memRepo != nil,
		VectorStore:      qdConn != nil,
		VectorCollection: ragBuild.CollectionReady,
		Embedding:        embedClient != nil,
		Rerank:           rerankClient != nil,
		ASR:              asrReady,
		VideoSummary:     videoSummaryReady,
		LLMConfig:        cfg.LLM,
	})

	// Register tools
	registerTools(registry, cfg, embedClient, rerankClient, asrClient, caps, ragBuild.Runtime)

	// Start MCP server if enabled
	if cfg.MCP.Enabled {
		mcpSrv := mcp.New(cfg.MCP.Addr, cfg.MCP.Mode, registry)
		mcpSrv.StartAsync()
	}

	// Build and start Gin engine
	e := server.Router(cfg, registry, ragBuild.Runtime, chatRepo, memRepo, caps)
	httpServer, err := server.NewHTTPServer(cfg.Server, e)
	if err != nil {
		panic(fmt.Errorf("build http server: %w", err))
	}
	slog.Info("gateway.starting", "addr", cfg.Server.Addr,
		"qdrant", qdConn != nil,
		"embedding", embedClient != nil,
		"rerank", rerankClient != nil,
		"asr", asrReady,
		"video_summary", videoSummaryReady,
		"mysql", chatRepo != nil,
		"memory", memRepo != nil,
		"mcp_enabled", cfg.MCP.Enabled,
	)

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
		shutdownTimeout, err := cfg.Server.ShutdownTimeoutDuration()
		if err != nil {
			shutdownTimeout = 10 * time.Second
		}
		slog.Info("gateway.shutting_down", "signal", sig.String(), "timeout", shutdownTimeout)
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), shutdownTimeout)
		defer shutdownCancel()
		if err := httpServer.Shutdown(shutdownCtx); err != nil {
			slog.Error("gateway.shutdown_failed", "err", err)
			_ = httpServer.Close()
		}
		if err := <-errCh; err != nil && !errors.Is(err, http.ErrServerClosed) {
			panic(err)
		}
	}

	if qdConn != nil {
		_ = qdConn.Close()
	}
}

func registerTools(registry *tool.Registry, cfg appconfig.Config, embedClient *model.EmbedClient, rerankClient *model.RerankClient, asrClient *asr.Client, caps capability.Snapshot, ragRuntime ragruntime.Runtime) {
	// Download & audio extraction tools (no external dependencies)
	_, dlWrap, err := tool.NewDownloadTool()
	if err != nil {
		slog.Warn("gateway.tool_register_failed", "tool", "download_video", "err", err)
	} else {
		registry.Register("download_video", dlWrap, nil)
	}

	_, downloadAudioWrap, err := tool.NewAudioDownloadTool()
	if err != nil {
		slog.Warn("gateway.tool_register_failed", "tool", "download_audio", "err", err)
	} else {
		registry.Register("download_audio", downloadAudioWrap, nil)
	}

	_, audioWrap, err := tool.NewAudioExtractTool()
	if err != nil {
		slog.Warn("gateway.tool_register_failed", "tool", "extract_audio", "err", err)
	} else {
		registry.Register("extract_audio", audioWrap, nil)
	}

	// Text format tool
	_, formatWrap, err := tool.NewTextFormatTool(cfg.LLM)
	if err != nil {
		slog.Warn("gateway.tool_register_failed", "tool", "format_transcript", "err", err)
	} else {
		registry.Register("format_transcript", formatWrap, nil)
	}

	if asrClient != nil {
		_, asrWrap, err := tool.NewASRTool(asrClient)
		if err != nil {
			slog.Warn("gateway.tool_register_failed", "tool", "transcribe_audio", "err", err)
		} else {
			registry.Register("transcribe_audio", asrWrap, nil)
		}
	}

	// RAG tools (require Qdrant + embedding)
	if caps.Available(capability.Embedding) && embedClient != nil {
		_, embWrap, err := tool.NewEmbedTool(embedClient)
		if err != nil {
			slog.Warn("gateway.tool_register_failed", "tool", "embed_texts", "err", err)
		} else {
			registry.Register("embed_texts", embWrap, nil)
		}
	}

	if caps.Available(capability.Rerank) && rerankClient != nil {
		_, rerankWrap, err := tool.NewRerankTool(rerankClient)
		if err != nil {
			slog.Warn("gateway.tool_register_failed", "tool", "rerank_documents", "err", err)
		} else {
			registry.Register("rerank_documents", rerankWrap, nil)
		}
	}

	if ragRuntime.Usable() {
		_, ragIdxWrap, err := tool.NewRAGIndexToolWithManager(ragRuntime.Sources)
		if err != nil {
			slog.Warn("gateway.tool_register_failed", "tool", "rag_index", "err", err)
		} else {
			registry.Register("rag_index", ragIdxWrap, nil)
		}

		_, ragQueryWrap, err := tool.NewRAGQueryTool(ragRuntime.Retriever)
		if err != nil {
			slog.Warn("gateway.tool_register_failed", "tool", "rag_query", "err", err)
		} else {
			registry.Register("rag_query", ragQueryWrap, nil)
		}

		_, ragListSourcesWrap, err := tool.NewRAGListSourcesToolWithManager(ragRuntime.Sources)
		if err != nil {
			slog.Warn("gateway.tool_register_failed", "tool", "rag_list_sources", "err", err)
		} else {
			registry.Register("rag_list_sources", ragListSourcesWrap, nil)
		}

		_, ragDeleteWrap, err := tool.NewRAGDeleteToolWithManager(ragRuntime.Sources)
		if err != nil {
			slog.Warn("gateway.tool_register_failed", "tool", "rag_delete", "err", err)
		} else {
			registry.Register("rag_delete", ragDeleteWrap, nil)
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
