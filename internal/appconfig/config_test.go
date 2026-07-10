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

func TestApplyDefaultsSetsServerTimeoutDefaults(t *testing.T) {
	var cfg Config
	cfg.applyDefaults()

	if cfg.Server.ReadHeaderTimeout != "5s" {
		t.Fatalf("ReadHeaderTimeout = %q, want 5s", cfg.Server.ReadHeaderTimeout)
	}
	if cfg.Server.ReadTimeout != "30s" {
		t.Fatalf("ReadTimeout = %q, want 30s", cfg.Server.ReadTimeout)
	}
	if cfg.Server.WriteTimeout != "0s" {
		t.Fatalf("WriteTimeout = %q, want 0s", cfg.Server.WriteTimeout)
	}
	if cfg.Server.IdleTimeout != "120s" {
		t.Fatalf("IdleTimeout = %q, want 120s", cfg.Server.IdleTimeout)
	}
	if cfg.Server.ShutdownTimeout != "10s" {
		t.Fatalf("ShutdownTimeout = %q, want 10s", cfg.Server.ShutdownTimeout)
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

func TestServerConfigDurations(t *testing.T) {
	cfg := ServerConfig{
		ReadHeaderTimeout: "3s",
		ReadTimeout:       "45s",
		WriteTimeout:      "0s",
		IdleTimeout:       "2m",
		ShutdownTimeout:   "15s",
	}

	if got, err := cfg.ReadHeaderTimeoutDuration(); err != nil || got != 3*time.Second {
		t.Fatalf("ReadHeaderTimeoutDuration = %s, %v; want 3s", got, err)
	}
	if got, err := cfg.ReadTimeoutDuration(); err != nil || got != 45*time.Second {
		t.Fatalf("ReadTimeoutDuration = %s, %v; want 45s", got, err)
	}
	if got, err := cfg.WriteTimeoutDuration(); err != nil || got != 0 {
		t.Fatalf("WriteTimeoutDuration = %s, %v; want 0s", got, err)
	}
	if got, err := cfg.IdleTimeoutDuration(); err != nil || got != 2*time.Minute {
		t.Fatalf("IdleTimeoutDuration = %s, %v; want 2m", got, err)
	}
	if got, err := cfg.ShutdownTimeoutDuration(); err != nil || got != 15*time.Second {
		t.Fatalf("ShutdownTimeoutDuration = %s, %v; want 15s", got, err)
	}
}

func TestValidateRejectsInvalidServerConfig(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Config)
		wantErr string
	}{
		{
			name: "invalid read header timeout",
			mutate: func(cfg *Config) {
				cfg.Server.ReadHeaderTimeout = "slow"
			},
			wantErr: "server.read_header_timeout",
		},
		{
			name: "zero read header timeout",
			mutate: func(cfg *Config) {
				cfg.Server.ReadHeaderTimeout = "0s"
			},
			wantErr: "server.read_header_timeout",
		},
		{
			name: "negative read timeout",
			mutate: func(cfg *Config) {
				cfg.Server.ReadTimeout = "-1s"
			},
			wantErr: "server.read_timeout",
		},
		{
			name: "negative write timeout",
			mutate: func(cfg *Config) {
				cfg.Server.WriteTimeout = "-1s"
			},
			wantErr: "server.write_timeout",
		},
		{
			name: "zero idle timeout",
			mutate: func(cfg *Config) {
				cfg.Server.IdleTimeout = "0s"
			},
			wantErr: "server.idle_timeout",
		},
		{
			name: "zero shutdown timeout",
			mutate: func(cfg *Config) {
				cfg.Server.ShutdownTimeout = "0s"
			},
			wantErr: "server.shutdown_timeout",
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
