package rag

import "context"

// SourceRegistry records the durable lifecycle state for indexed RAG sources.
// Vector payloads remain the retrieval source, while this registry is the
// operational/audit view used by user-facing source management.
type SourceRegistry interface {
	RecordIndexed(ctx context.Context, sources []SourceSummary) error
	MarkDeleted(ctx context.Context, req DeleteRequest) error
	ListSources(ctx context.Context, req SourceListRequest) ([]SourceSummary, error)
}
