package ragruntime

import (
	"context"
	"errors"
	"testing"

	"github.com/raiki02/vidwise/internal/appconfig"
	"github.com/raiki02/vidwise/internal/model"
	"github.com/raiki02/vidwise/internal/rag"
	qdrantclient "github.com/raiki02/vidwise/internal/storage/qdrant"
)

func TestBuildSkipsRuntimeWithoutVectorStoreOrEmbedding(t *testing.T) {
	cfg := testRuntimeConfig()

	tests := []struct {
		name string
		deps Deps
	}{
		{name: "missing qdrant", deps: Deps{Embed: &model.EmbedClient{}}},
		{name: "missing embedding", deps: Deps{Qdrant: &qdrantclient.Client{}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Build(context.Background(), cfg, tt.deps)
			if got.CollectionReady {
				t.Fatal("expected collection not ready")
			}
			if got.Runtime.Usable() {
				t.Fatalf("expected runtime unusable, got %#v", got.Runtime)
			}
			if got.Err != nil {
				t.Fatalf("expected no ensure error when adapters are missing, got %v", got.Err)
			}
		})
	}
}

func TestBuildAssemblesRuntimeWithNoReranker(t *testing.T) {
	cfg := testRuntimeConfig()
	got := Build(context.Background(), cfg, Deps{
		Qdrant:           &qdrantclient.Client{},
		Embed:            &model.EmbedClient{},
		EnsureCollection: noopEnsureCollection,
	})

	if got.Err != nil {
		t.Fatalf("Build error: %v", got.Err)
	}
	if !got.CollectionReady {
		t.Fatal("expected collection ready")
	}
	if !got.Runtime.Usable() {
		t.Fatalf("expected usable runtime, got %#v", got.Runtime)
	}
	if got.Runtime.Indexer == nil {
		t.Fatal("expected indexer")
	}
	if got.Runtime.Retriever == nil {
		t.Fatal("expected retriever")
	}
	if got.Runtime.Collection != "test_chunks" {
		t.Fatalf("Collection = %q, want test_chunks", got.Runtime.Collection)
	}
}

func TestBuildAttachesSourceRegistryToRuntime(t *testing.T) {
	cfg := testRuntimeConfig()
	registry := &fakeSourceRegistry{}
	got := Build(context.Background(), cfg, Deps{
		Qdrant:           &qdrantclient.Client{},
		Embed:            &model.EmbedClient{},
		Registry:         registry,
		EnsureCollection: noopEnsureCollection,
	})

	if got.Err != nil {
		t.Fatalf("Build error: %v", got.Err)
	}
	if got.Runtime.Registry != registry {
		t.Fatalf("registry not attached to runtime")
	}
	if got.Runtime.Catalog == nil {
		t.Fatal("expected source catalog")
	}
	if got.Runtime.Sources == nil || !got.Runtime.Sources.CanIndex() || !got.Runtime.Sources.CanList() {
		t.Fatalf("expected source manager with index/list support, got %#v", got.Runtime.Sources)
	}
}

func TestBuildStopsRuntimeWhenCollectionEnsureFails(t *testing.T) {
	cfg := testRuntimeConfig()
	wantErr := errors.New("collection down")
	got := Build(context.Background(), cfg, Deps{
		Qdrant: &qdrantclient.Client{},
		Embed:  &model.EmbedClient{},
		EnsureCollection: func(context.Context, *rag.Indexer) error {
			return wantErr
		},
	})

	if !errors.Is(got.Err, wantErr) {
		t.Fatalf("Err = %v, want %v", got.Err, wantErr)
	}
	if got.CollectionReady {
		t.Fatal("expected collection not ready")
	}
	if got.Runtime.Usable() {
		t.Fatalf("expected runtime unusable after ensure failure, got %#v", got.Runtime)
	}
}

func TestConfigMapping(t *testing.T) {
	cfg := testRuntimeConfig()

	retrieval := RetrieverConfig(cfg)
	contextCfg := ContextConfig(cfg)

	if retrieval.SearchTopK != 32 || retrieval.TopK != 12 || retrieval.MinScore != 0.25 {
		t.Fatalf("unexpected retrieval config: %#v", retrieval)
	}
	if contextCfg.MaxRunes != 4096 {
		t.Fatalf("unexpected context config: %#v", contextCfg)
	}
}

func noopEnsureCollection(context.Context, *rag.Indexer) error {
	return nil
}

type fakeSourceRegistry struct{}

func (*fakeSourceRegistry) RecordIndexed(context.Context, []rag.SourceSummary) error { return nil }
func (*fakeSourceRegistry) MarkDeleted(context.Context, rag.DeleteRequest) error     { return nil }
func (*fakeSourceRegistry) ListSources(context.Context, rag.SourceListRequest) ([]rag.SourceSummary, error) {
	return nil, nil
}

func testRuntimeConfig() appconfig.Config {
	enabled := false
	return appconfig.Config{
		LLM: appconfig.LLMConfig{Enabled: &enabled},
		Qdrant: appconfig.QdrantConfig{
			Collection: "test_chunks",
		},
		RAG: appconfig.RAGConfig{
			Retrieval: appconfig.RAGRetrievalConfig{
				SearchTopK: 32,
				TopK:       12,
				MinScore:   0.25,
			},
			Context: appconfig.RAGContextConfig{
				MaxRunes: 4096,
			},
		},
	}
}
