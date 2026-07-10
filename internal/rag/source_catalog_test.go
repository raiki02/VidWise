package rag

import (
	"context"
	"errors"
	"testing"

	pb "github.com/qdrant/go-client/qdrant"
	qdrantclient "github.com/raiki02/vidwise/internal/storage/qdrant"
	"google.golang.org/grpc"
)

type fakeSourceRegistry struct {
	sources []SourceSummary
	err     error
	calls   int
}

func (r *fakeSourceRegistry) RecordIndexed(context.Context, []SourceSummary) error {
	return nil
}

func (r *fakeSourceRegistry) MarkDeleted(context.Context, DeleteRequest) error {
	return nil
}

func (r *fakeSourceRegistry) ListSources(context.Context, SourceListRequest) ([]SourceSummary, error) {
	r.calls++
	if r.err != nil {
		return nil, r.err
	}
	return r.sources, nil
}

type fakeSourceScroller struct {
	requests  []*pb.ScrollPoints
	responses []*pb.ScrollResponse
	err       error
}

func (s *fakeSourceScroller) Scroll(ctx context.Context, req *pb.ScrollPoints, opts ...grpc.CallOption) (*pb.ScrollResponse, error) {
	s.requests = append(s.requests, req)
	if s.err != nil {
		return nil, s.err
	}
	if len(s.responses) == 0 {
		return &pb.ScrollResponse{}, nil
	}
	resp := s.responses[0]
	s.responses = s.responses[1:]
	return resp, nil
}

func TestSourceCatalogListSourcesAggregatesChunks(t *testing.T) {
	scroller := &fakeSourceScroller{
		responses: []*pb.ScrollResponse{{
			Result: []*pb.RetrievedPoint{
				{Payload: sourcePayload("source-1", "Guide", "guide.md", "markdown", "u1", "s1")},
				{Payload: sourcePayload("source-1", "Guide", "guide.md", "markdown", "u1", "s1")},
				{Payload: sourcePayload("source-2", "Notes", "notes.md", "markdown", "u1", "s1")},
			},
		}},
	}
	catalog := newSourceCatalogWithAdapters(scroller, nil, "rag_docs", 100)

	got, err := catalog.ListSources(context.Background(), SourceListRequest{
		Filter: &RetrieveFilter{UserID: "u1", SessionID: "s1"},
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListSources: %v", err)
	}

	if len(got) != 2 {
		t.Fatalf("expected two sources, got %#v", got)
	}
	if got[0].SourceID != "source-1" || got[0].ChunkCount != 2 || got[0].DocumentTitle != "Guide" {
		t.Fatalf("unexpected first source: %#v", got[0])
	}
	if got[1].SourceID != "source-2" || got[1].ChunkCount != 1 {
		t.Fatalf("unexpected second source: %#v", got[1])
	}
}

func TestSourceCatalogListSourcesScansAllPagesBeforeApplyingLimit(t *testing.T) {
	scroller := &fakeSourceScroller{
		responses: []*pb.ScrollResponse{
			{
				Result: []*pb.RetrievedPoint{
					{Payload: sourcePayload("source-1", "Guide", "guide.md", "markdown", "u1", "s1")},
				},
				NextPageOffset: &pb.PointId{PointIdOptions: &pb.PointId_Num{Num: 2}},
			},
			{
				Result: []*pb.RetrievedPoint{
					{Payload: sourcePayload("source-1", "Guide", "guide.md", "markdown", "u1", "s1")},
				},
			},
		},
	}
	catalog := newSourceCatalogWithAdapters(scroller, nil, "rag_docs", 1)

	got, err := catalog.ListSources(context.Background(), SourceListRequest{
		Filter: &RetrieveFilter{UserID: "u1"},
		Limit:  1,
	})
	if err != nil {
		t.Fatalf("ListSources: %v", err)
	}
	if len(scroller.requests) != 2 {
		t.Fatalf("expected both pages to be scanned, got %d requests", len(scroller.requests))
	}
	if len(got) != 1 || got[0].ChunkCount != 2 {
		t.Fatalf("expected complete chunk count after pagination, got %#v", got)
	}
}

func TestBuildSourceCatalogScrollRequestUsesScopeAndPayloadOnly(t *testing.T) {
	pageSize := uint32(32)
	req := buildSourceCatalogScrollRequest("rag_docs", pageSize, nil, buildScopeFilter(&RetrieveFilter{
		UserID:    "u1",
		SessionID: "s1",
	}))

	if req.CollectionName != "rag_docs" {
		t.Fatalf("collection = %q, want rag_docs", req.CollectionName)
	}
	if req.GetLimit() != pageSize {
		t.Fatalf("limit = %d, want %d", req.GetLimit(), pageSize)
	}
	filter := req.GetFilter()
	if filter == nil || len(filter.GetMust()) != 2 {
		t.Fatalf("expected user/session filter, got %#v", filter)
	}
	fields := req.GetWithPayload().GetInclude().GetFields()
	if !contains(fields, qdrantclient.FieldSourceID) || !contains(fields, qdrantclient.FieldDocumentTitle) {
		t.Fatalf("expected source payload fields, got %#v", fields)
	}
	if req.GetWithVectors().GetEnable() {
		t.Fatal("source catalog must not load vectors")
	}
}

func TestSourceCatalogUsesRegistryWhenAvailable(t *testing.T) {
	registry := &fakeSourceRegistry{
		sources: []SourceSummary{{SourceID: "source-1", SourceName: "guide.md", ChunkCount: 2}},
	}
	scroller := &fakeSourceScroller{}
	catalog := newSourceCatalogWithAdapters(scroller, registry, "rag_docs", 1)

	got, err := catalog.ListSources(context.Background(), SourceListRequest{Filter: &RetrieveFilter{UserID: "u1"}})
	if err != nil {
		t.Fatalf("ListSources: %v", err)
	}
	if len(got) != 1 || got[0].SourceID != "source-1" {
		t.Fatalf("unexpected registry sources: %#v", got)
	}
	if registry.calls != 1 {
		t.Fatalf("registry calls = %d, want 1", registry.calls)
	}
	if len(scroller.requests) != 0 {
		t.Fatalf("expected registry result to avoid Qdrant scan, got %d requests", len(scroller.requests))
	}
}

func TestSourceCatalogFallsBackToQdrantWhenRegistryIsEmpty(t *testing.T) {
	registry := &fakeSourceRegistry{}
	scroller := &fakeSourceScroller{
		responses: []*pb.ScrollResponse{{
			Result: []*pb.RetrievedPoint{
				{Payload: sourcePayload("source-1", "Guide", "guide.md", "markdown", "u1", "s1")},
			},
		}},
	}
	catalog := newSourceCatalogWithAdapters(scroller, registry, "rag_docs", 100)

	got, err := catalog.ListSources(context.Background(), SourceListRequest{Filter: &RetrieveFilter{UserID: "u1"}})
	if err != nil {
		t.Fatalf("ListSources: %v", err)
	}
	if len(got) != 1 || got[0].SourceID != "source-1" {
		t.Fatalf("expected Qdrant fallback sources, got %#v", got)
	}
	if len(scroller.requests) != 1 {
		t.Fatalf("expected Qdrant fallback scan, got %d requests", len(scroller.requests))
	}
}

func TestSourceCatalogRejectsMissingScroller(t *testing.T) {
	_, err := newSourceCatalogWithAdapters(nil, nil, "rag_docs", 0).ListSources(context.Background(), SourceListRequest{})
	if err == nil {
		t.Fatal("expected missing catalog error")
	}
}

func TestSourceCatalogReturnsScrollError(t *testing.T) {
	wantErr := errors.New("qdrant down")
	_, err := newSourceCatalogWithAdapters(&fakeSourceScroller{err: wantErr}, nil, "rag_docs", 0).ListSources(context.Background(), SourceListRequest{})
	if !errors.Is(err, wantErr) {
		t.Fatalf("expected wrapped scroll error, got %v", err)
	}
}

func sourcePayload(sourceID, title, sourceName, contentType, userID, sessionID string) map[string]*pb.Value {
	return map[string]*pb.Value{
		qdrantclient.FieldSourceID:      stringPayload(sourceID),
		qdrantclient.FieldDocumentTitle: stringPayload(title),
		qdrantclient.FieldSourceName:    stringPayload(sourceName),
		qdrantclient.FieldContentType:   stringPayload(contentType),
		qdrantclient.FieldUserID:        stringPayload(userID),
		qdrantclient.FieldSessionID:     stringPayload(sessionID),
	}
}

func contains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
