package rag

import (
	"context"
	"errors"
	"testing"
)

type fakeLifecycleIndexer struct {
	indexResult  IndexResult
	deleteResult DeleteResult
	indexErr     error
	deleteErr    error
	indexCalls   int
	deleteCalls  int
	indexSource  Source
	indexOptions IndexOptions
	indexUserID  string
	indexSession string
	deleteReq    DeleteRequest
}

func (i *fakeLifecycleIndexer) IndexSourceScoped(_ context.Context, source Source, opts IndexOptions, userID, sessionID string) (IndexResult, error) {
	i.indexCalls++
	i.indexSource = source
	i.indexOptions = opts
	i.indexUserID = userID
	i.indexSession = sessionID
	if i.indexErr != nil {
		return IndexResult{}, i.indexErr
	}
	return i.indexResult, nil
}

func (i *fakeLifecycleIndexer) DeleteSourcesWithOptions(_ context.Context, req DeleteRequest) (DeleteResult, error) {
	i.deleteCalls++
	i.deleteReq = req
	if i.deleteErr != nil {
		return DeleteResult{}, i.deleteErr
	}
	return i.deleteResult, nil
}

type fakeLifecycleCatalog struct {
	sources []SourceSummary
	err     error
	calls   int
	req     SourceListRequest
}

func (c *fakeLifecycleCatalog) ListSources(_ context.Context, req SourceListRequest) ([]SourceSummary, error) {
	c.calls++
	c.req = req
	if c.err != nil {
		return nil, c.err
	}
	return c.sources, nil
}

type fakeLifecycleRegistry struct {
	recordErr error
	deleteErr error
	indexed   []SourceSummary
	deleted   DeleteRequest
}

func (r *fakeLifecycleRegistry) RecordIndexed(_ context.Context, sources []SourceSummary) error {
	r.indexed = append(r.indexed, sources...)
	return r.recordErr
}

func (r *fakeLifecycleRegistry) MarkDeleted(_ context.Context, req DeleteRequest) error {
	r.deleted = req
	return r.deleteErr
}

func (r *fakeLifecycleRegistry) ListSources(context.Context, SourceListRequest) ([]SourceSummary, error) {
	return nil, nil
}

func TestSourceManagerIndexRecordsRegistry(t *testing.T) {
	indexer := &fakeLifecycleIndexer{
		indexResult: IndexResult{
			ChunkCount:  2,
			ContentType: "markdown",
			SourceIDs:   []string{"source-1"},
			Sources:     []SourceSummary{{SourceID: "source-1", ChunkCount: 2}},
		},
	}
	registry := &fakeLifecycleRegistry{}
	manager := newSourceManagerWithAdapters(indexer, nil, registry)

	got, err := manager.IndexSourceScoped(context.Background(), Source{
		Text:     "# Guide",
		Filename: "guide.md",
	}, IndexOptions{ChunkRunes: 256}, "user-1", "session-1")
	if err != nil {
		t.Fatalf("IndexSourceScoped: %v", err)
	}

	if got.ChunkCount != 2 || len(got.SourceIDs) != 1 {
		t.Fatalf("unexpected index result: %#v", got)
	}
	if indexer.indexCalls != 1 || indexer.indexUserID != "user-1" || indexer.indexSession != "session-1" {
		t.Fatalf("indexer call not preserved: %#v", indexer)
	}
	if len(registry.indexed) != 1 || registry.indexed[0].SourceID != "source-1" {
		t.Fatalf("registry not recorded: %#v", registry.indexed)
	}
}

func TestSourceManagerIndexIgnoresRegistryErrorAfterSuccessfulIndex(t *testing.T) {
	indexer := &fakeLifecycleIndexer{
		indexResult: IndexResult{
			SourceIDs: []string{"source-1"},
			Sources:   []SourceSummary{{SourceID: "source-1"}},
		},
	}
	registry := &fakeLifecycleRegistry{recordErr: errors.New("mysql down")}
	manager := newSourceManagerWithAdapters(indexer, nil, registry)

	got, err := manager.IndexSourceScoped(context.Background(), Source{Text: "hello"}, IndexOptions{}, "user-1", "")
	if err != nil {
		t.Fatalf("IndexSourceScoped should return successful index despite registry error: %v", err)
	}
	if len(got.SourceIDs) != 1 || len(registry.indexed) != 1 {
		t.Fatalf("unexpected result or registry attempt: result=%#v registry=%#v", got, registry.indexed)
	}
}

func TestSourceManagerDeleteMarksRegistry(t *testing.T) {
	indexer := &fakeLifecycleIndexer{deleteResult: DeleteResult{SourceIDs: []string{"source-1"}}}
	registry := &fakeLifecycleRegistry{}
	manager := newSourceManagerWithAdapters(indexer, nil, registry)
	req := DeleteRequest{
		SourceIDs: []string{"source-1"},
		Filter:    &RetrieveFilter{UserID: "user-1"},
	}

	got, err := manager.DeleteSourcesWithOptions(context.Background(), req)
	if err != nil {
		t.Fatalf("DeleteSourcesWithOptions: %v", err)
	}

	if len(got.SourceIDs) != 1 || got.SourceIDs[0] != "source-1" {
		t.Fatalf("unexpected delete result: %#v", got)
	}
	if indexer.deleteCalls != 1 || indexer.deleteReq.Filter == nil || indexer.deleteReq.Filter.UserID != "user-1" {
		t.Fatalf("indexer delete request not preserved: %#v", indexer.deleteReq)
	}
	if len(registry.deleted.SourceIDs) != 1 || registry.deleted.SourceIDs[0] != "source-1" {
		t.Fatalf("registry delete not recorded: %#v", registry.deleted)
	}
}

func TestSourceManagerListDelegatesToCatalog(t *testing.T) {
	catalog := &fakeLifecycleCatalog{sources: []SourceSummary{{SourceID: "source-1"}}}
	manager := newSourceManagerWithAdapters(nil, catalog, nil)

	got, err := manager.ListSources(context.Background(), SourceListRequest{
		Filter: &RetrieveFilter{SessionID: "session-1"},
		Limit:  5,
	})
	if err != nil {
		t.Fatalf("ListSources: %v", err)
	}
	if len(got) != 1 || got[0].SourceID != "source-1" {
		t.Fatalf("unexpected sources: %#v", got)
	}
	if catalog.calls != 1 || catalog.req.Limit != 5 || catalog.req.Filter.SessionID != "session-1" {
		t.Fatalf("catalog request not preserved: %#v", catalog.req)
	}
}

func TestSourceManagerRejectsMissingAdapters(t *testing.T) {
	if _, err := (*SourceManager)(nil).IndexSourceScoped(context.Background(), Source{}, IndexOptions{}, "", ""); err == nil {
		t.Fatal("expected missing indexer error")
	}
	if _, err := (*SourceManager)(nil).DeleteSourcesWithOptions(context.Background(), DeleteRequest{}); err == nil {
		t.Fatal("expected missing indexer error")
	}
	if _, err := (*SourceManager)(nil).ListSources(context.Background(), SourceListRequest{}); err == nil {
		t.Fatal("expected missing catalog error")
	}
}
