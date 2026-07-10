package ragruntime

import (
	"context"

	"github.com/raiki02/vidwise/internal/appconfig"
	"github.com/raiki02/vidwise/internal/model"
	"github.com/raiki02/vidwise/internal/rag"
	qdrantclient "github.com/raiki02/vidwise/internal/storage/qdrant"
)

// Runtime is the assembled RAG module graph for one gateway process.
type Runtime struct {
	Indexer    *rag.Indexer
	Retriever  *rag.Retriever
	Catalog    *rag.SourceCatalog
	Sources    *rag.SourceManager
	Registry   rag.SourceRegistry
	Retrieval  rag.RetrieverConfig
	Context    rag.ContextConfig
	Collection string
}

func (r Runtime) Usable() bool {
	return r.Indexer != nil && r.Retriever != nil
}

// Deps are the concrete adapters available during gateway assembly.
type Deps struct {
	Qdrant   *qdrantclient.Client
	Embed    *model.EmbedClient
	Rerank   *model.RerankClient
	Registry rag.SourceRegistry

	EnsureCollection func(context.Context, *rag.Indexer) error
}

type BuildResult struct {
	Runtime         Runtime
	CollectionReady bool
	Err             error
}

func Build(ctx context.Context, cfg appconfig.Config, deps Deps) BuildResult {
	runtime := Runtime{
		Retrieval:  RetrieverConfig(cfg),
		Context:    ContextConfig(cfg),
		Collection: cfg.Qdrant.Collection,
	}

	if deps.Qdrant == nil || deps.Embed == nil {
		return BuildResult{Runtime: runtime}
	}

	indexer := rag.NewIndexer(deps.Embed, deps.Qdrant, cfg.Qdrant.Collection)
	ensure := deps.EnsureCollection
	if ensure == nil {
		ensure = func(ctx context.Context, idx *rag.Indexer) error {
			return idx.EnsureCollection(ctx)
		}
	}
	if err := ensure(ctx, indexer); err != nil {
		return BuildResult{
			Runtime: runtime,
			Err:     err,
		}
	}

	runtime.Indexer = indexer
	runtime.Retriever = rag.NewRetrieverWithConfig(deps.Embed, deps.Rerank, deps.Qdrant, cfg.Qdrant.Collection, runtime.Retrieval)
	runtime.Registry = deps.Registry
	runtime.Catalog = rag.NewSourceCatalogWithRegistry(deps.Qdrant, cfg.Qdrant.Collection, deps.Registry)
	runtime.Sources = rag.NewSourceManager(indexer, runtime.Catalog, deps.Registry)
	return BuildResult{
		Runtime:         runtime,
		CollectionReady: true,
	}
}

func RetrieverConfig(cfg appconfig.Config) rag.RetrieverConfig {
	return rag.RetrieverConfig{
		SearchTopK: cfg.RAG.Retrieval.SearchTopK,
		TopK:       cfg.RAG.Retrieval.TopK,
		MinScore:   cfg.RAG.Retrieval.MinScore,
	}
}

func ContextConfig(cfg appconfig.Config) rag.ContextConfig {
	return rag.ContextConfig{
		MaxRunes: cfg.RAG.Context.MaxRunes,
	}
}
