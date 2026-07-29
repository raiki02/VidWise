package search

import (
	"context"
	"fmt"
	"net/url"
	"strings"
	"time"
)

// SearchService is the application boundary used by Eino tools. It intentionally
// hides provider, crawler, rerank, and compression details from the Agent layer.
type SearchService interface {
	Search(ctx context.Context, query string) (*SearchResult, error)
}

type QueryRewriter interface {
	Rewrite(ctx context.Context, query string) ([]string, error)
}

type SearchProvider interface {
	Search(ctx context.Context, query string) ([]SearchItem, error)
}

type Crawler interface {
	Fetch(ctx context.Context, urls []string) ([]Document, error)
}

type Extractor interface {
	Extract(ctx context.Context, docs []Document) ([]Document, error)
}

type Reranker interface {
	Rank(query string, docs []Document) []Document
}

type ContextualReranker interface {
	RankWithContext(ctx context.Context, query string, docs []Document) ([]Document, error)
}

type Compressor interface {
	Compress(docs []Document) []Document
}

type QualityEvaluator interface {
	Evaluate(query string, docs []Document) SearchQuality
}

type SearchCache interface {
	Get(query string) (*SearchResult, bool)
	Set(query string, result SearchResult)
}

type Metrics interface {
	ObserveSearch(ctx context.Context, status string, elapsed time.Duration)
	ObserveProvider(ctx context.Context, provider string, status string, elapsed time.Duration)
	ObserveCache(ctx context.Context, hit bool)
}

type noopMetrics struct{}

func (noopMetrics) ObserveSearch(context.Context, string, time.Duration)           {}
func (noopMetrics) ObserveProvider(context.Context, string, string, time.Duration) {}
func (noopMetrics) ObserveCache(context.Context, bool)                             {}

type MockProvider struct {
	Name    ProviderName
	Results map[string][]SearchItem
	Default []SearchItem
}

func NewMockProvider(results map[string][]SearchItem) *MockProvider {
	return &MockProvider{
		Name:    ProviderMock,
		Results: results,
	}
}

func (p *MockProvider) Search(ctx context.Context, query string) ([]SearchItem, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("mock search canceled: %w", err)
	}
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, fmt.Errorf("mock search query is required")
	}

	if p != nil {
		if items := p.Results[query]; len(items) > 0 {
			return withProvider(items, providerNameOrDefault(p.Name, ProviderMock)), nil
		}
		if items := p.Results[strings.ToLower(query)]; len(items) > 0 {
			return withProvider(items, providerNameOrDefault(p.Name, ProviderMock)), nil
		}
		if len(p.Default) > 0 {
			return withProvider(p.Default, providerNameOrDefault(p.Name, ProviderMock)), nil
		}
	}

	escaped := url.QueryEscape(query)
	return []SearchItem{{
		Title:    "Mock search result: " + query,
		URL:      "https://mock.local/search?q=" + escaped,
		Snippet:  "Mock result for " + query + ". Replace MockProvider with Bing, Tavily, DuckDuckGo, or internal search in production.",
		Provider: ProviderMock,
	}}, nil
}

func withProvider(items []SearchItem, provider ProviderName) []SearchItem {
	out := make([]SearchItem, 0, len(items))
	for _, item := range items {
		if item.Provider == "" {
			item.Provider = provider
		}
		out = append(out, item)
	}
	return out
}

func providerNameOrDefault(name, fallback ProviderName) ProviderName {
	if name == "" {
		return fallback
	}
	return name
}
