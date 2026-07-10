package main

import (
	"context"
	"testing"

	"github.com/raiki02/vidwise/internal/appconfig"
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

	registerTools(registry, appconfig.Config{LLM: testMainLLMConfig()}, nil, nil, caps, ragruntime.Runtime{})

	assertWrappedTool(t, registry, "download_video")
	assertWrappedTool(t, registry, "extract_audio")
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
		caps,
		testMainRAGRuntime(caps, &model.RerankClient{}),
	)

	assertToolPresent(t, registry, "rerank_documents")
	assertWrappedTool(t, registry, "rerank_documents")
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
