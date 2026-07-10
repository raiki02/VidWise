package appconfig

import (
	"strings"
	"testing"
	"time"
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

func TestApplyDefaultsSetsTaskDefaults(t *testing.T) {
	var cfg Config
	cfg.applyDefaults()

	if cfg.Task.MaxTracked != 1000 {
		t.Fatalf("MaxTracked = %d, want 1000", cfg.Task.MaxTracked)
	}
	if cfg.Task.RetainFor != "24h" {
		t.Fatalf("RetainFor = %q, want 24h", cfg.Task.RetainFor)
	}
	retention, err := cfg.Task.RetentionDuration()
	if err != nil {
		t.Fatalf("RetentionDuration returned error: %v", err)
	}
	if retention != 24*time.Hour {
		t.Fatalf("RetentionDuration = %s, want 24h", retention)
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

func TestTaskConfigRetentionDuration(t *testing.T) {
	tests := []struct {
		name    string
		cfg     TaskConfig
		want    time.Duration
		wantErr string
	}{
		{
			name: "empty uses default",
			cfg:  TaskConfig{},
			want: 24 * time.Hour,
		},
		{
			name: "custom duration",
			cfg:  TaskConfig{RetainFor: "90m"},
			want: 90 * time.Minute,
		},
		{
			name:    "invalid duration",
			cfg:     TaskConfig{RetainFor: "forever"},
			wantErr: "invalid duration",
		},
		{
			name:    "non-positive duration",
			cfg:     TaskConfig{RetainFor: "0s"},
			wantErr: "greater than 0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.cfg.RetentionDuration()
			if tt.wantErr != "" {
				if err == nil {
					t.Fatal("expected validation error")
				}
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("expected error containing %q, got %v", tt.wantErr, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("RetentionDuration returned error: %v", err)
			}
			if got != tt.want {
				t.Fatalf("RetentionDuration = %s, want %s", got, tt.want)
			}
		})
	}
}

func TestValidateRejectsInvalidTaskConfig(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Config)
		wantErr string
	}{
		{
			name: "invalid max tracked",
			mutate: func(cfg *Config) {
				cfg.Task.MaxTracked = -1
			},
			wantErr: "task.max_tracked",
		},
		{
			name: "invalid retention duration",
			mutate: func(cfg *Config) {
				cfg.Task.RetainFor = "forever"
			},
			wantErr: "task.retain_for",
		},
		{
			name: "non-positive retention duration",
			mutate: func(cfg *Config) {
				cfg.Task.RetainFor = "0s"
			},
			wantErr: "task.retain_for",
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
