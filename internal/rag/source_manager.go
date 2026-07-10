package rag

import (
	"context"
	"errors"
	"log/slog"
)

type sourceIndexer interface {
	IndexSourceScoped(ctx context.Context, source Source, opts IndexOptions, userID, sessionID string) (IndexResult, error)
	DeleteSourcesWithOptions(ctx context.Context, req DeleteRequest) (DeleteResult, error)
}

type sourceCatalog interface {
	ListSources(ctx context.Context, req SourceListRequest) ([]SourceSummary, error)
}

// SourceManager owns the RAG source lifecycle: index, list, delete, and
// operational registry side effects.
type SourceManager struct {
	indexer  sourceIndexer
	catalog  sourceCatalog
	registry SourceRegistry
}

func NewSourceManager(indexer *Indexer, catalog *SourceCatalog, registry SourceRegistry) *SourceManager {
	var indexerAdapter sourceIndexer
	if indexer != nil {
		indexerAdapter = indexer
	}
	var catalogAdapter sourceCatalog
	if catalog != nil {
		catalogAdapter = catalog
	}
	return newSourceManagerWithAdapters(indexerAdapter, catalogAdapter, registry)
}

func newSourceManagerWithAdapters(indexer sourceIndexer, catalog sourceCatalog, registry SourceRegistry) *SourceManager {
	return &SourceManager{
		indexer:  indexer,
		catalog:  catalog,
		registry: registry,
	}
}

func (m *SourceManager) CanIndex() bool {
	return m != nil && m.indexer != nil
}

func (m *SourceManager) CanList() bool {
	return m != nil && m.catalog != nil
}

func (m *SourceManager) IndexSourceScoped(ctx context.Context, source Source, opts IndexOptions, userID, sessionID string) (IndexResult, error) {
	if m == nil || m.indexer == nil {
		return IndexResult{}, errors.New("rag source indexer is required")
	}
	result, err := m.indexer.IndexSourceScoped(ctx, source, opts, userID, sessionID)
	if err != nil {
		return IndexResult{}, err
	}
	m.recordIndexed(ctx, result.Sources)
	return result, nil
}

func (m *SourceManager) DeleteSourcesWithOptions(ctx context.Context, req DeleteRequest) (DeleteResult, error) {
	if m == nil || m.indexer == nil {
		return DeleteResult{}, errors.New("rag source indexer is required")
	}
	result, err := m.indexer.DeleteSourcesWithOptions(ctx, req)
	if err != nil {
		return DeleteResult{}, err
	}
	m.markDeleted(ctx, req)
	return result, nil
}

func (m *SourceManager) ListSources(ctx context.Context, req SourceListRequest) ([]SourceSummary, error) {
	if m == nil || m.catalog == nil {
		return nil, errors.New("rag source catalog is required")
	}
	return m.catalog.ListSources(ctx, req)
}

func (m *SourceManager) recordIndexed(ctx context.Context, sources []SourceSummary) {
	if m == nil || m.registry == nil || len(sources) == 0 {
		return
	}
	if err := m.registry.RecordIndexed(ctx, sources); err != nil {
		slog.Warn("rag.source_registry_record_failed", "err", err)
	}
}

func (m *SourceManager) markDeleted(ctx context.Context, req DeleteRequest) {
	if m == nil || m.registry == nil {
		return
	}
	if err := m.registry.MarkDeleted(ctx, req); err != nil {
		slog.Warn("rag.source_registry_delete_failed", "err", err)
	}
}
