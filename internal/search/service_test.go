package search

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestServiceSearchRunsPipelineAndCaches(t *testing.T) {
	provider := &recordingProvider{items: []SearchItem{{
		Title:   "Provider title",
		URL:     "https://docs.example/eino-search",
		Snippet: "provider snippet",
	}}}
	crawler := &recordingCrawler{docs: []Document{{
		URL:  "https://docs.example/eino-search",
		html: `<html><head><title>Eino Search Docs</title></head><body><nav>menu</nav><main>Use Eino tools for web search evidence.</main></body></html>`,
	}}}
	cache := NewMemoryCache(MemoryCacheConfig{})
	svc, err := NewService(ServiceDependencies{
		Cache: cache,
		Rewriter: MockQueryRewriter{Responses: map[string][]string{
			"最近OpenAI有什么消息": {"OpenAI latest announcement"},
		}},
		Router: NewProviderRouter(ProviderRegistration{
			Name:     ProviderMock,
			Provider: provider,
		}),
		Crawler:    crawler,
		Extractor:  NewBasicExtractor(),
		Reranker:   NewKeywordReranker(),
		Compressor: NewBasicCompressor(BasicCompressorConfig{MaxDocuments: 3, MaxContentRunes: 200}),
	})
	if err != nil {
		t.Fatalf("NewService() error = %v", err)
	}

	result, err := svc.Search(context.Background(), "最近OpenAI有什么消息")
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(provider.queries) != 2 || provider.queries[0] != "OpenAI latest announcement" || provider.queries[1] != "最近OpenAI有什么消息" {
		t.Fatalf("provider queries = %#v", provider.queries)
	}
	if len(crawler.urls) != 1 || crawler.urls[0] != "https://docs.example/eino-search" {
		t.Fatalf("crawler urls = %#v", crawler.urls)
	}
	if len(result.Sources) != 1 {
		t.Fatalf("sources len = %d, want 1", len(result.Sources))
	}
	if result.Sources[0].Title != "Eino Search Docs" {
		t.Fatalf("source title = %q", result.Sources[0].Title)
	}
	if strings.Contains(result.Sources[0].Content, "menu") || !strings.Contains(result.Sources[0].Content, "web search evidence") {
		t.Fatalf("unexpected extracted content = %q", result.Sources[0].Content)
	}

	cached, err := svc.Search(context.Background(), "最近OpenAI有什么消息")
	if err != nil {
		t.Fatalf("cached Search() error = %v", err)
	}
	if !cached.Cached {
		t.Fatal("second result should be marked cached")
	}
	if len(provider.queries) != 2 {
		t.Fatalf("provider called after cache hit: %#v", provider.queries)
	}
}

func TestServiceFallsBackToProviderSnippetsWhenCrawlFails(t *testing.T) {
	provider := &recordingProvider{items: []SearchItem{{
		Title:   "Snippet source",
		URL:     "https://news.example/openai",
		Snippet: "OpenAI announced a new feature.",
	}}}
	svc, err := NewService(ServiceDependencies{
		Router: NewProviderRouter(ProviderRegistration{Name: ProviderMock, Provider: provider}),
		Crawler: &recordingCrawler{
			err: errors.New("network down"),
		},
		Extractor:  NewBasicExtractor(),
		Reranker:   NewKeywordReranker(),
		Compressor: NewBasicCompressor(BasicCompressorConfig{}),
	})
	if err != nil {
		t.Fatalf("NewService() error = %v", err)
	}

	result, err := svc.Search(context.Background(), "OpenAI feature")
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(result.Sources) != 1 {
		t.Fatalf("sources len = %d, want 1", len(result.Sources))
	}
	if result.Sources[0].Content != "OpenAI announced a new feature." {
		t.Fatalf("fallback content = %q", result.Sources[0].Content)
	}
}

func TestServiceRejectsEmptyQuery(t *testing.T) {
	svc, err := NewMockService()
	if err != nil {
		t.Fatalf("NewMockService() error = %v", err)
	}
	if _, err := svc.Search(context.Background(), "  "); err == nil {
		t.Fatal("expected empty query error")
	}
}

type recordingProvider struct {
	items   []SearchItem
	err     error
	queries []string
}

func (p *recordingProvider) Search(_ context.Context, query string) ([]SearchItem, error) {
	p.queries = append(p.queries, query)
	if p.err != nil {
		return nil, p.err
	}
	return p.items, nil
}

type recordingCrawler struct {
	docs []Document
	err  error
	urls []string
}

func (c *recordingCrawler) Fetch(_ context.Context, urls []string) ([]Document, error) {
	c.urls = append(c.urls, urls...)
	if c.err != nil {
		return nil, c.err
	}
	return c.docs, nil
}
