package search

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/raiki02/vidwise/internal/rag"
)

type InternalProviderConfig struct {
	SearchTopK int
	TopK       int
	MinScore   float64
}

type InternalSearchProvider struct {
	retriever interface {
		RetrieveWithOptions(context.Context, rag.RetrieveRequest) ([]rag.RelevantChunk, error)
	}
	cfg InternalProviderConfig
}

func NewInternalSearchProvider(retriever interface {
	RetrieveWithOptions(context.Context, rag.RetrieveRequest) ([]rag.RelevantChunk, error)
}, cfg InternalProviderConfig) (*InternalSearchProvider, error) {
	if retriever == nil {
		return nil, errors.New("internal search retriever is required")
	}
	if cfg.SearchTopK <= 0 {
		cfg.SearchTopK = 20
	}
	if cfg.TopK <= 0 {
		cfg.TopK = 8
	}
	return &InternalSearchProvider{retriever: retriever, cfg: cfg}, nil
}

func (p *InternalSearchProvider) Search(ctx context.Context, query string) ([]SearchItem, error) {
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, errors.New("internal search query is required")
	}
	minScore := p.cfg.MinScore
	chunks, err := p.retriever.RetrieveWithOptions(ctx, rag.RetrieveRequest{
		Query:      query,
		SearchTopK: p.cfg.SearchTopK,
		TopK:       p.cfg.TopK,
		MinScore:   &minScore,
	})
	if err != nil {
		return nil, fmt.Errorf("retrieve internal knowledge: %w", err)
	}
	items := make([]SearchItem, 0, len(chunks))
	for _, chunk := range chunks {
		title := firstNonEmpty(chunk.DocumentTitle, chunk.SourceName, chunk.SourceID, "Internal knowledge")
		url := firstNonEmpty(chunk.SourceURL, internalSearchURL(chunk))
		items = append(items, SearchItem{
			Title:    title,
			URL:      url,
			Snippet:  chunk.Text,
			Provider: ProviderInternal,
		})
	}
	return items, nil
}

func internalSearchURL(chunk rag.RelevantChunk) string {
	if chunk.SourceID == "" && chunk.ChunkID == "" {
		return ""
	}
	id := firstNonEmpty(chunk.ChunkID, chunk.SourceID)
	return "internal://rag/" + id
}
