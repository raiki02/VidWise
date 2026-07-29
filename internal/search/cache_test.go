package search

import (
	"testing"
	"time"
)

func TestMemoryCacheReturnsClonedResultAndExpires(t *testing.T) {
	now := time.Date(2026, 7, 28, 12, 0, 0, 0, time.UTC)
	cache := NewMemoryCache(MemoryCacheConfig{
		TTL: time.Minute,
		Now: func() time.Time {
			return now
		},
	})
	cache.Set(" OpenAI ", SearchResult{
		Summary: "summary",
		Sources: []Source{{Title: "one", URL: "https://example.com", Content: "content"}},
	})

	got, ok := cache.Get("openai")
	if !ok {
		t.Fatal("expected cache hit")
	}
	got.Sources[0].Content = "mutated"

	gotAgain, ok := cache.Get("OPENAI")
	if !ok {
		t.Fatal("expected second cache hit")
	}
	if gotAgain.Sources[0].Content != "content" {
		t.Fatalf("cached result mutated: %q", gotAgain.Sources[0].Content)
	}

	now = now.Add(2 * time.Minute)
	if _, ok := cache.Get("openai"); ok {
		t.Fatal("expected expired cache miss")
	}
}
