package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	"github.com/raiki02/vidwise/internal/tool"
)

// Server wraps the mcp-go MCPServer and bridges it with the tool registry.
type Server struct {
	srv      *server.MCPServer
	sse      *server.SSEServer
	registry *tool.Registry
	addr     string
}

// New creates a new MCP server that registers all tools from the registry.
func New(addr, mode string, registry *tool.Registry) *Server {
	s := &Server{
		registry: registry,
		addr:     addr,
	}

	mcpServer := server.NewMCPServer(
		"video-extractor",
		"1.0.0",
		server.WithToolCapabilities(true),
	)

	// Register all tools from the registry as MCP tools
	for name, entry := range registry.List() {
		mcpTool := mcp.Tool{
			Name:        name,
			Description: generateDescription(name),
			InputSchema: mcp.ToolInputSchema{
				Type:       "object",
				Properties: map[string]any{},
			},
		}

		// Capture entry in loop
		toolEntry := entry
		mcpServer.AddTool(mcpTool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			argsMap := req.GetArguments()
			argsJSON := "{}"
			if len(argsMap) > 0 {
				if b, err := json.Marshal(argsMap); err == nil {
					argsJSON = string(b)
				}
			}
			result, err := toolEntry.Tool.InvokableRun(ctx, argsJSON)
			if err != nil {
				return mcp.NewToolResultError(err.Error()), nil
			}
			return mcp.NewToolResultText(result), nil
		})
		slog.Info("mcp.tool_registered", "name", name)
	}

	s.srv = mcpServer
	if mode == "sse" {
		s.sse = server.NewSSEServer(mcpServer)
	}
	return s
}

// Start starts the MCP server in SSE mode.
func (s *Server) Start() error {
	if s.sse == nil {
		return fmt.Errorf("SSE server not initialized (only SSE mode supported)")
	}
	slog.Info("mcp.server.starting", "addr", s.addr)
	return s.sse.Start(s.addr)
}

// StartAsync starts the MCP server in a background goroutine.
func (s *Server) StartAsync() {
	go func() {
		if err := s.Start(); err != nil {
			slog.Error("mcp.server.failed", "err", err)
		}
	}()
}

func generateDescription(name string) string {
	descriptions := map[string]string{
		"download_video":    "Download a video from a URL using yt-dlp.",
		"extract_audio":     "Extract audio from a video URL using yt-dlp.",
		"transcribe_audio":  "Transcribe a local audio file to text using the ASR service.",
		"summarize_video":   "Summarize a local video by calling the video understanding service.",
		"format_transcript": "Format raw ASR transcript text using an LLM with typo fixing and paragraph organization.",
		"embed_texts":       "Generate vector embeddings for text strings using the configured embedding model.",
		"rerank_documents":  "Rerank a list of documents by relevance to a query.",
		"rag_index":         "Index text into the RAG knowledge base (Qdrant vector database).",
		"rag_query":         "Search the RAG knowledge base for relevant context chunks.",
	}
	if desc, ok := descriptions[name]; ok {
		return desc
	}
	return fmt.Sprintf("Tool: %s", name)
}
