package search

import (
	"context"
	"strings"
	"testing"
)

func TestProviderRouterPrefersNewsProvidersWhenRegistered(t *testing.T) {
	router := NewProviderRouter(
		ProviderRegistration{Name: ProviderMock, Provider: NewMockProvider(nil)},
		ProviderRegistration{Name: ProviderBing, Provider: NewMockProvider(nil)},
		ProviderRegistration{Name: ProviderInternal, Provider: NewMockProvider(nil)},
	)

	providers, err := router.Route(context.Background(), "OpenAI latest news")
	if err != nil {
		t.Fatalf("Route() error = %v", err)
	}
	if len(providers) < 2 {
		t.Fatalf("providers len = %d", len(providers))
	}
	if providers[0].Name != ProviderBing {
		t.Fatalf("first provider = %q, want bing", providers[0].Name)
	}
}

func TestKeywordRerankerRanksKeywordMatchesFirst(t *testing.T) {
	reranker := NewKeywordReranker()
	docs := reranker.Rank("OpenAI search", []Document{
		{Title: "Cooking", Content: strings.Repeat("recipe ", 80)},
		{Title: "OpenAI Search", Content: "OpenAI search announcement"},
	})

	if docs[0].Title != "OpenAI Search" {
		t.Fatalf("top doc = %q", docs[0].Title)
	}
}

func TestBasicCompressorLimitsDocumentsAndContent(t *testing.T) {
	compressor := NewBasicCompressor(BasicCompressorConfig{
		MaxDocuments:    1,
		MaxContentRunes: 12,
		MaxTotalRunes:   12,
	})
	docs := compressor.Compress([]Document{
		{Title: "one", Content: "abcdefghijklmnopqrstuvwxyz"},
		{Title: "two", Content: "second"},
	})

	if len(docs) != 1 {
		t.Fatalf("docs len = %d, want 1", len(docs))
	}
	if len([]rune(docs[0].Content)) > 12 {
		t.Fatalf("compressed content too long: %q", docs[0].Content)
	}
	if !strings.HasSuffix(docs[0].Content, "...") {
		t.Fatalf("expected truncation suffix, got %q", docs[0].Content)
	}
}
