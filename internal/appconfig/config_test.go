package appconfig

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestLoadDefaults(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte(`
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: test-key
qdrant:
  vector_dim: 1024
`), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Server.Addr != ":8080" {
		t.Errorf("server.addr = %q, want :8080", cfg.Server.Addr)
	}
	if cfg.Download.WorkDir != ".vidwise/media" {
		t.Errorf("download.work_dir = %q", cfg.Download.WorkDir)
	}
	if cfg.ASR.BaseURL != "http://localhost:8001" {
		t.Errorf("asr.base_url = %q", cfg.ASR.BaseURL)
	}
	if cfg.ASR.Transcribe.BeamSize != 5 {
		t.Errorf("asr.transcribe.beam_size = %d", cfg.ASR.Transcribe.BeamSize)
	}
	if cfg.Embedding.BaseURL != "http://localhost:8003" {
		t.Errorf("embedding.base_url = %q", cfg.Embedding.BaseURL)
	}
	if cfg.Qdrant.Addr() != "localhost:6334" {
		t.Errorf("qdrant addr = %q", cfg.Qdrant.Addr())
	}
	if cfg.Qdrant.Collection != "transcript_chunks" {
		t.Errorf("qdrant.collection = %q", cfg.Qdrant.Collection)
	}
	if cfg.RAG.ChunkRunes != 800 || cfg.RAG.OverlapRunes != 100 || cfg.RAG.TopK != 5 {
		t.Errorf("rag defaults = %+v", cfg.RAG)
	}
	if cfg.Agent.MaxStep != 20 || cfg.Agent.HistoryLimit != 20 {
		t.Errorf("agent defaults = %+v", cfg.Agent)
	}
	readHeader, err := cfg.Server.ReadHeaderTimeoutDuration()
	if err != nil || readHeader != 5*time.Second {
		t.Errorf("read_header_timeout = %v, %v", readHeader, err)
	}
	writeTimeout, err := cfg.Server.WriteTimeoutDuration()
	if err != nil || writeTimeout != 0 {
		t.Errorf("write_timeout = %v, %v (must stay 0)", writeTimeout, err)
	}
}

func TestLoadRequiresLLM(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte("qdrant:\n  vector_dim: 1024\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err == nil {
		t.Fatal("expected error for missing llm.model")
	}
}

func TestLoadRequiresQdrantDim(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte("llm:\n  provider: openai\n  model: m\n  api_key: k\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err == nil {
		t.Fatal("expected error for missing qdrant.vector_dim")
	}
}

func TestLLMResolvedAPIKey(t *testing.T) {
	cfg := LLMConfig{APIKey: "explicit"}
	if got := cfg.ResolvedAPIKey(); got != "explicit" {
		t.Errorf("ResolvedAPIKey = %q", got)
	}
	cfg = LLMConfig{APIKeyEnv: "VIDWISE_TEST_API_KEY"}
	if got := cfg.ResolvedAPIKey(); got != "" {
		t.Errorf("ResolvedAPIKey = %q, want empty", got)
	}
	t.Setenv("VIDWISE_TEST_API_KEY", "from-env")
	if got := cfg.ResolvedAPIKey(); got != "from-env" {
		t.Errorf("ResolvedAPIKey = %q, want from-env", got)
	}
}

func TestLLMKeepAliveDuration(t *testing.T) {
	cfg := LLMConfig{KeepAlive: ""}
	d, err := cfg.KeepAliveDuration()
	if err != nil || d != nil {
		t.Errorf("empty keep_alive = %v, %v", d, err)
	}
	cfg = LLMConfig{KeepAlive: "5m"}
	d, err = cfg.KeepAliveDuration()
	if err != nil || d == nil || *d != 5*time.Minute {
		t.Errorf("5m keep_alive = %v, %v", d, err)
	}
}
