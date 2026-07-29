package search

import (
	"context"
	"fmt"
	"sort"
)

type RerankResult struct {
	Index int
	Score float64
	Text  string
}

type RerankClient interface {
	Rerank(ctx context.Context, query string, documents []string) ([]RerankResult, error)
}

type EmbeddingReranker struct {
	client   RerankClient
	fallback Reranker
}

func NewEmbeddingReranker(client RerankClient, fallback Reranker) (*EmbeddingReranker, error) {
	if client == nil {
		return nil, fmt.Errorf("embedding rerank client is required")
	}
	if fallback == nil {
		fallback = NewKeywordReranker()
	}
	return &EmbeddingReranker{client: client, fallback: fallback}, nil
}

func (r *EmbeddingReranker) Rank(query string, docs []Document) []Document {
	return r.fallback.Rank(query, docs)
}

func (r *EmbeddingReranker) RankWithContext(ctx context.Context, query string, docs []Document) ([]Document, error) {
	if len(docs) <= 1 {
		return append([]Document(nil), docs...), nil
	}
	contents := make([]string, 0, len(docs))
	for _, doc := range docs {
		contents = append(contents, doc.Title+"\n"+doc.Content)
	}
	results, err := r.client.Rerank(ctx, query, contents)
	if err != nil {
		return nil, fmt.Errorf("embedding rerank documents: %w", err)
	}
	out := make([]Document, 0, len(results))
	used := map[int]struct{}{}
	for _, result := range results {
		if result.Index < 0 || result.Index >= len(docs) {
			continue
		}
		doc := docs[result.Index]
		doc.Score = result.Score
		out = append(out, doc)
		used[result.Index] = struct{}{}
	}
	if len(out) == 0 {
		return r.fallback.Rank(query, docs), nil
	}
	sort.SliceStable(out, func(i, j int) bool {
		return out[i].Score > out[j].Score
	})
	for i, doc := range docs {
		if _, ok := used[i]; ok {
			continue
		}
		out = append(out, doc)
	}
	return out, nil
}
