package search

import (
	"context"
	"testing"

	"github.com/raiki02/vidwise/internal/rag"
)

func TestInternalSearchProviderMapsRAGChunks(t *testing.T) {
	provider, err := NewInternalSearchProvider(fakeInternalRetriever{
		chunks: []rag.RelevantChunk{{
			Text:          "internal knowledge chunk",
			DocumentTitle: "Guide",
			SourceURL:     "https://kb.example/guide",
		}},
	}, InternalProviderConfig{SearchTopK: 5, TopK: 2})
	if err != nil {
		t.Fatalf("NewInternalSearchProvider() error = %v", err)
	}

	items, err := provider.Search(context.Background(), "guide")
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("items len = %d, want 1", len(items))
	}
	if items[0].Provider != ProviderInternal || items[0].Title != "Guide" || items[0].Snippet != "internal knowledge chunk" {
		t.Fatalf("unexpected item: %#v", items[0])
	}
}

type fakeInternalRetriever struct {
	chunks []rag.RelevantChunk
	err    error
}

func (r fakeInternalRetriever) RetrieveWithOptions(_ context.Context, _ rag.RetrieveRequest) ([]rag.RelevantChunk, error) {
	return r.chunks, r.err
}
