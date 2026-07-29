package main

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/raiki02/vidwise/internal/appconfig"
	"github.com/raiki02/vidwise/internal/asr"
	"github.com/raiki02/vidwise/internal/capability"
	"github.com/raiki02/vidwise/internal/model"
	"github.com/raiki02/vidwise/internal/rag"
	"github.com/raiki02/vidwise/internal/ragruntime"
	qdrantclient "github.com/raiki02/vidwise/internal/storage/qdrant"
	"github.com/raiki02/vidwise/internal/tool"
)

func TestRegisterToolsDoesNotExposeRAGToolsWhenRAGUnavailable(t *testing.T) {
	registry := tool.NewRegistry()
	caps := capability.FromRuntime(capability.RuntimeDeps{LLMConfig: testMainLLMConfig()})

	registerTools(registry, appconfig.Config{LLM: testMainLLMConfig()}, nil, nil, testMainASRClient(t), caps, ragruntime.Runtime{})

	assertWrappedTool(t, registry, "download_video")
	assertWrappedTool(t, registry, "download_audio")
	assertWrappedTool(t, registry, "extract_audio")
	assertWrappedTool(t, registry, "transcribe_audio")
	assertWrappedTool(t, registry, "format_transcript")
	assertToolMissing(t, registry, "embed_texts")
	assertToolMissing(t, registry, "rerank_documents")
	assertToolMissing(t, registry, "rag_index")
	assertToolMissing(t, registry, "rag_query")
}

func TestRegisterToolsExposesRAGToolsWhenRAGIsDegradedByMissingRerank(t *testing.T) {
	registry := tool.NewRegistry()
	caps := capability.FromRuntime(capability.RuntimeDeps{
		VectorStore:      true,
		VectorCollection: true,
		Embedding:        true,
		Rerank:           false,
		LLMConfig:        testMainLLMConfig(),
	})

	registerTools(
		registry,
		appconfig.Config{
			LLM:    testMainLLMConfig(),
			Qdrant: appconfig.QdrantConfig{Collection: "test_chunks"},
		},
		&model.EmbedClient{},
		nil,
		testMainASRClient(t),
		caps,
		testMainRAGRuntime(caps, nil),
	)

	assertToolPresent(t, registry, "embed_texts")
	assertToolMissing(t, registry, "rerank_documents")
	assertToolPresent(t, registry, "rag_index")
	assertToolPresent(t, registry, "rag_query")
	assertWrappedTool(t, registry, "embed_texts")
	assertWrappedTool(t, registry, "rag_index")
	assertWrappedTool(t, registry, "rag_query")
}

func TestRegisterToolsExposesRerankToolWhenRerankAvailable(t *testing.T) {
	registry := tool.NewRegistry()
	caps := capability.FromRuntime(capability.RuntimeDeps{
		VectorStore:      true,
		VectorCollection: true,
		Embedding:        true,
		Rerank:           true,
		LLMConfig:        testMainLLMConfig(),
	})

	registerTools(
		registry,
		appconfig.Config{
			LLM:    testMainLLMConfig(),
			Qdrant: appconfig.QdrantConfig{Collection: "test_chunks"},
		},
		&model.EmbedClient{},
		&model.RerankClient{},
		testMainASRClient(t),
		caps,
		testMainRAGRuntime(caps, &model.RerankClient{}),
	)

	assertToolPresent(t, registry, "rerank_documents")
	assertWrappedTool(t, registry, "rerank_documents")
}

func TestRegisterToolsExposesWebSearchWhenEnabledWithMockProvider(t *testing.T) {
	registry := tool.NewRegistry()
	caps := capability.FromRuntime(capability.RuntimeDeps{LLMConfig: testMainLLMConfig()})
	cfg := appconfig.Config{
		LLM:    testMainLLMConfig(),
		Search: testMainSearchConfig("mock"),
	}

	registerTools(registry, cfg, nil, nil, testMainASRClient(t), caps, ragruntime.Runtime{})

	assertWrappedTool(t, registry, "web_search")
}

func TestNewSearchServiceRejectsBingWithoutAPIKey(t *testing.T) {
	cfg := testMainSearchConfig("bing")
	cfg.Bing.APIKey = ""
	cfg.Bing.APIKeyEnv = "VIDWISE_TEST_EMPTY_BING_KEY"

	_, err := newSearchService(cfg, testMainLLMConfig(), nil, ragruntime.Runtime{})
	if err == nil || !strings.Contains(err.Error(), "bing") {
		t.Fatalf("newSearchService error = %v, want bing key error", err)
	}
}

func TestNewSearchServiceRejectsEmbeddingRerankWithoutClient(t *testing.T) {
	cfg := testMainSearchConfig("mock")
	cfg.RerankProvider = "embedding"

	_, err := newSearchService(cfg, testMainLLMConfig(), nil, ragruntime.Runtime{})
	if err == nil || !strings.Contains(err.Error(), "rerank") {
		t.Fatalf("newSearchService error = %v, want rerank client error", err)
	}
}

func assertToolPresent(t *testing.T, registry *tool.Registry, name string) {
	t.Helper()
	if _, err := registry.Get(name); err != nil {
		t.Fatalf("expected tool %q to be registered: %v", name, err)
	}
}

func assertToolMissing(t *testing.T, registry *tool.Registry, name string) {
	t.Helper()
	if _, err := registry.Get(name); err == nil {
		t.Fatalf("expected tool %q to be absent", name)
	}
}

func assertWrappedTool(t *testing.T, registry *tool.Registry, name string) {
	t.Helper()
	got, err := registry.Get(name)
	if err != nil {
		t.Fatalf("expected tool %q to be registered: %v", name, err)
	}
	if _, ok := got.(*tool.Wrapper); !ok {
		t.Fatalf("expected tool %q to be *tool.Wrapper, got %T", name, got)
	}
}

func testMainLLMConfig() appconfig.LLMConfig {
	enabled := false
	return appconfig.LLMConfig{Enabled: &enabled}
}

func testMainSearchConfig(providers ...string) appconfig.SearchConfig {
	if len(providers) == 0 {
		providers = []string{"mock"}
	}
	return appconfig.SearchConfig{
		Enabled:                true,
		Provider:               providers[0],
		Providers:              providers,
		Timeout:                "10s",
		QueryRewriteProvider:   "mock",
		QueryRewriteMaxQueries: 3,
		RerankProvider:         "keyword",
		CacheProvider:          "memory",
		CacheTTL:               "5m",
		MaxCacheKeys:           1024,
		MaxResults:             10,
		MaxDocuments:           5,
		MaxContentRunes:        1200,
		MaxTotalRunes:          6000,
		MaxResponseBytes:       2 * 1024 * 1024,
		MaxConcurrency:         4,
		UserAgent:              "test-agent",
		Bing: appconfig.SearchProviderConfig{
			BaseURL:    "https://api.bing.microsoft.com/v7.0/search",
			APIKey:     "test-key",
			MaxResults: 10,
		},
		Tavily: appconfig.SearchProviderConfig{
			BaseURL:     "https://api.tavily.com/search",
			APIKey:      "test-key",
			MaxResults:  10,
			SearchDepth: "basic",
		},
		DuckDuckGo: appconfig.SearchProviderConfig{
			BaseURL:    "https://duckduckgo.com/html/",
			MaxResults: 10,
		},
		Internal: appconfig.SearchInternalProviderConfig{
			SearchTopK: 20,
			TopK:       8,
		},
		Redis: appconfig.SearchRedisConfig{
			Addr:      "localhost:6379",
			KeyPrefix: "vidwise:search:",
			Timeout:   "1s",
		},
	}
}

func testMainASRClient(t *testing.T) *asr.Client {
	t.Helper()
	client, err := asr.NewClient("http://127.0.0.1:8001", "zh", time.Minute, asr.TranscribeOptions{})
	if err != nil {
		t.Fatalf("build asr client: %v", err)
	}
	return client
}

func testMainRAGRuntime(caps capability.Snapshot, reranker *model.RerankClient) ragruntime.Runtime {
	return ragruntime.Build(context.Background(), appconfig.Config{
		Qdrant: appconfig.QdrantConfig{Collection: "test_chunks"},
	}, ragruntime.Deps{
		Qdrant:           &qdrantclient.Client{},
		Embed:            &model.EmbedClient{},
		Rerank:           reranker,
		EnsureCollection: func(context.Context, *rag.Indexer) error { return nil },
	}).Runtime
}
