package tool

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
	"github.com/raiki02/vidwise/internal/rag"
)

// RAGIndexInput is the input for the RAG indexing tool.
type RAGIndexInput struct {
	Text         string            `json:"text" jsonschema:"required" jsonschema_description:"The text to parse, chunk, embed, and index into the vector database."`
	Filename     string            `json:"filename,omitempty" jsonschema_description:"Optional source filename. Markdown extensions enable heading-aware parsing."`
	ContentType  string            `json:"content_type,omitempty" jsonschema_description:"Optional MIME content type, such as text/markdown."`
	Format       string            `json:"format,omitempty" jsonschema_description:"Optional source format: auto, plain, or markdown."`
	Metadata     map[string]string `json:"metadata,omitempty" jsonschema_description:"Optional source metadata to attach to indexed chunks."`
	UserID       string            `json:"user_id,omitempty" jsonschema_description:"User id that scopes indexed chunks. Required unless session_id is provided."`
	SessionID    string            `json:"session_id,omitempty" jsonschema_description:"Session id that scopes indexed chunks. Required unless user_id is provided."`
	ChunkRunes   int               `json:"chunk_runes,omitempty" jsonschema_description:"Optional max runes per chunk for this indexing call."`
	OverlapRunes *int              `json:"overlap_runes,omitempty" jsonschema_description:"Optional sliding-window overlap runes for this indexing call."`
}

type RAGIndexOutput struct {
	ChunkCount  int                 `json:"chunk_count"`
	ContentType string              `json:"content_type"`
	SourceIDs   []string            `json:"source_ids,omitempty"`
	Sources     []rag.SourceSummary `json:"sources,omitempty"`
}

type ragIndexRequest struct {
	Source    rag.Source
	Options   rag.IndexOptions
	UserID    string
	SessionID string
}

func NewRAGIndexTool(indexer *rag.Indexer) (tool.InvokableTool, *Wrapper, error) {
	return NewRAGIndexToolWithRegistry(indexer, nil)
}

func NewRAGIndexToolWithRegistry(indexer *rag.Indexer, registry rag.SourceRegistry) (tool.InvokableTool, *Wrapper, error) {
	return NewRAGIndexToolWithManager(rag.NewSourceManager(indexer, nil, registry))
}

func NewRAGIndexToolWithManager(manager *rag.SourceManager) (tool.InvokableTool, *Wrapper, error) {
	if manager == nil || !manager.CanIndex() {
		return nil, nil, errors.New("rag source manager is required")
	}
	inner, err := utils.InferTool(
		"rag_index",
		"Index text or Markdown into the RAG knowledge base within an explicit user_id or session_id scope. Parses structured sources, enriches metadata, chunks, embeds, and stores them in the Qdrant vector database.",
		func(ctx context.Context, input RAGIndexInput) (RAGIndexOutput, error) {
			req, err := normalizeRAGIndexInput(input)
			if err != nil {
				return RAGIndexOutput{}, err
			}
			result, err := manager.IndexSourceScoped(ctx, req.Source, req.Options, req.UserID, req.SessionID)
			if err != nil {
				return RAGIndexOutput{}, err
			}
			return RAGIndexOutput{
				ChunkCount:  result.ChunkCount,
				ContentType: result.ContentType,
				SourceIDs:   result.SourceIDs,
				Sources:     result.Sources,
			}, nil
		},
	)
	if err != nil {
		return nil, nil, err
	}
	wrapper := NewWrapper(inner, WrapperConfig{Name: "rag_index", Timeout: 0})
	return inner, wrapper, nil
}

func normalizeRAGIndexInput(input RAGIndexInput) (ragIndexRequest, error) {
	text := strings.TrimSpace(input.Text)
	if text == "" {
		return ragIndexRequest{}, errors.New("text is required")
	}

	format, err := parseRAGContentFormat(input.Format)
	if err != nil {
		return ragIndexRequest{}, err
	}
	scope, err := rag.ResolveScope(input.UserID, input.SessionID, rag.StrictScopePolicy())
	if err != nil {
		return ragIndexRequest{}, err
	}

	return ragIndexRequest{
		Source: rag.Source{
			Text:        text,
			Filename:    strings.TrimSpace(input.Filename),
			ContentType: strings.TrimSpace(input.ContentType),
			Format:      format,
			Metadata:    input.Metadata,
		},
		Options: rag.IndexOptions{
			ChunkRunes:   input.ChunkRunes,
			OverlapRunes: input.OverlapRunes,
		},
		UserID:    scope.UserID,
		SessionID: scope.SessionID,
	}, nil
}

func parseRAGContentFormat(format string) (rag.ContentFormat, error) {
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "", "auto":
		return rag.ContentFormatAuto, nil
	case "plain", "text":
		return rag.ContentFormatPlain, nil
	case "markdown", "md":
		return rag.ContentFormatMarkdown, nil
	default:
		return rag.ContentFormatAuto, fmt.Errorf("format must be one of: auto, plain, markdown")
	}
}
