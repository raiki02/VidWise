package appconfig

import (
	"strings"
	"testing"
)

func TestApplyDefaultsSetsRAGRetrievalDefaults(t *testing.T) {
	var cfg Config
	cfg.applyDefaults()

	if cfg.RAG.Retrieval.SearchTopK != 20 {
		t.Fatalf("SearchTopK = %d, want 20", cfg.RAG.Retrieval.SearchTopK)
	}
	if cfg.RAG.Retrieval.TopK != 8 {
		t.Fatalf("TopK = %d, want 8", cfg.RAG.Retrieval.TopK)
	}
	if cfg.RAG.Retrieval.MinScore != 0 {
		t.Fatalf("MinScore = %f, want 0", cfg.RAG.Retrieval.MinScore)
	}
	if cfg.RAG.Context.MaxRunes != 12000 {
		t.Fatalf("MaxRunes = %d, want 12000", cfg.RAG.Context.MaxRunes)
	}
}

func TestValidateRejectsInvalidRAGRetrievalConfig(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Config)
		wantErr string
	}{
		{
			name: "top k greater than search top k",
			mutate: func(cfg *Config) {
				cfg.RAG.Retrieval.SearchTopK = 5
				cfg.RAG.Retrieval.TopK = 6
			},
			wantErr: "rag.retrieval.top_k",
		},
		{
			name: "negative min score",
			mutate: func(cfg *Config) {
				cfg.RAG.Retrieval.MinScore = -0.1
			},
			wantErr: "rag.retrieval.min_score",
		},
		{
			name: "invalid context budget",
			mutate: func(cfg *Config) {
				cfg.RAG.Context.MaxRunes = 0
			},
			wantErr: "rag.context.max_runes",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var cfg Config
			cfg.applyDefaults()
			tt.mutate(&cfg)

			err := cfg.validate()
			if err == nil {
				t.Fatal("expected validation error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("expected error containing %q, got %v", tt.wantErr, err)
			}
		})
	}
}
