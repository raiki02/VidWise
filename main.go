package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/raiki02/video-extractor/internal/appconfig"
	"github.com/raiki02/video-extractor/internal/mcp"
	"github.com/raiki02/video-extractor/internal/server"
	"github.com/raiki02/video-extractor/internal/tool"
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

	// Start MCP server if enabled
	if cfg.MCP.Enabled {
		mcpSrv := mcp.New(cfg.MCP.Addr, cfg.MCP.Mode, registry)
		mcpSrv.StartAsync()
	}

	// Build and start Gin engine
	e := server.Router(cfg, registry)
	slog.Info("gateway.starting", "addr", cfg.Server.Addr)

	// Graceful shutdown
	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
		<-sig
		slog.Info("gateway.shutting_down")
		os.Exit(0)
	}()

	if err := e.Run(cfg.Server.Addr); err != nil {
		panic(err)
	}
}

func runWorker(cfg appconfig.Config) {
	slog.Info("worker.starting")
	slog.Info("worker.waiting_for_tasks")

	// In production: poll MySQL for pending tasks and execute them
	// For now, the worker is a stub that can be extended with task execution logic

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig
	slog.Info("worker.shutting_down")
}
