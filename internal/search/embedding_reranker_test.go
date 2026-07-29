package search

import (
	"context"
	"testing"
)

func TestEmbeddingRerankerUsesExternalScores(t *testing.T) {
	reranker, err := NewEmbeddingReranker(fakeRerankClient{
		results: []RerankResult{
			{Index: 1, Score: 0.9},
			{Index: 0, Score: 0.4},
		},
	}, NewKeywordReranker())
	if err != nil {
		t.Fatalf("NewEmbeddingReranker() error = %v", err)
	}

	docs, err := reranker.RankWithContext(context.Background(), "query", []Document{
		{Title: "first", Content: "a"},
		{Title: "second", Content: "b"},
	})
	if err != nil {
		t.Fatalf("RankWithContext() error = %v", err)
	}
	if docs[0].Title != "second" || docs[0].Score != 0.9 {
		t.Fatalf("unexpected ranked docs: %#v", docs)
	}
}

type fakeRerankClient struct {
	results []RerankResult
	err     error
}

func (c fakeRerankClient) Rerank(context.Context, string, []string) ([]RerankResult, error) {
	return c.results, c.err
}
