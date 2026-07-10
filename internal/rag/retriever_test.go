package rag

import (
	"context"
	"errors"
	"strings"
	"testing"

	pb "github.com/qdrant/go-client/qdrant"
	"github.com/raiki02/vidwise/internal/model"
	qdrantclient "github.com/raiki02/vidwise/internal/storage/qdrant"
)

func TestNewRetrieveFilterNormalizesEmptyScope(t *testing.T) {
	if got := NewRetrieveFilter(" ", "\t"); got != nil {
		t.Fatalf("expected nil filter for empty scope, got %#v", got)
	}

	got := NewRetrieveFilter(" user-1 ", " session-1 ")
	if got == nil {
		t.Fatal("expected filter")
	}
	if got.UserID != "user-1" {
		t.Fatalf("UserID = %q, want user-1", got.UserID)
	}
	if got.SessionID != "session-1" {
		t.Fatalf("SessionID = %q, want session-1", got.SessionID)
	}
}

func TestBuildScopeFilterIncludesUserAndSession(t *testing.T) {
	filter := buildScopeFilter(NewRetrieveFilter("user-1", "session-1"))
	if filter == nil {
		t.Fatal("expected qdrant filter")
	}
	if len(filter.Must) != 2 {
		t.Fatalf("expected 2 must clauses, got %#v", filter.Must)
	}

	got := map[string]string{}
	for _, cond := range filter.Must {
		field := cond.GetField()
		if field == nil || field.Match == nil {
			t.Fatalf("unexpected condition: %#v", cond)
		}
		got[field.Key] = field.Match.GetKeyword()
	}

	if got[qdrantclient.FieldUserID] != "user-1" {
		t.Fatalf("user filter missing: %#v", got)
	}
	if got[qdrantclient.FieldSessionID] != "session-1" {
		t.Fatalf("session filter missing: %#v", got)
	}
}

func TestNormalizeRetrieveRequestAppliesDefaultsAndClamp(t *testing.T) {
	retriever := &Retriever{
		searchTopK: 20,
		topK:       8,
		minScore:   0.4,
	}

	got := retriever.normalizeRetrieveRequest(RetrieveRequest{
		Query:      "  hello  ",
		SearchTopK: 5,
		TopK:       10,
		Filter:     &RetrieveFilter{UserID: " u1 "},
	})

	if got.Query != "hello" {
		t.Fatalf("Query = %q, want hello", got.Query)
	}
	if got.SearchTopK != 5 {
		t.Fatalf("SearchTopK = %d, want 5", got.SearchTopK)
	}
	if got.TopK != 5 {
		t.Fatalf("TopK = %d, want clamp to 5", got.TopK)
	}
	if got.Filter == nil || got.Filter.UserID != "u1" {
		t.Fatalf("filter not normalized: %#v", got.Filter)
	}
	if got.MinScore == nil || *got.MinScore != 0.4 {
		t.Fatalf("MinScore = %v, want default 0.4", got.MinScore)
	}
}

func TestNormalizeRetrieveRequestAllowsPerCallMinScoreToDisableDefault(t *testing.T) {
	retriever := &Retriever{
		searchTopK: 20,
		topK:       8,
		minScore:   0.5,
	}

	disabled := 0.0
	got := retriever.normalizeRetrieveRequest(RetrieveRequest{
		Query:    "hello",
		MinScore: &disabled,
	})

	if got.MinScore == nil || *got.MinScore != 0 {
		t.Fatalf("MinScore = %v, want explicit 0", got.MinScore)
	}
}

func TestRetrieveWithOptionsRejectsEmptyQueryBeforeExternalCalls(t *testing.T) {
	retriever := &Retriever{
		searchTopK: 20,
		topK:       8,
	}

	_, err := retriever.RetrieveWithOptions(context.Background(), RetrieveRequest{Query: "  "})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestRetrieveWithOptionsReturnsEmbedError(t *testing.T) {
	retriever := newRetrieverWithAdapters(
		&fakeQueryEmbedder{err: errors.New("embed down")},
		nil,
		&fakeVectorSearcher{},
		"chunks",
		DefaultRetrieverConfig(),
	)

	_, err := retriever.RetrieveWithOptions(context.Background(), RetrieveRequest{Query: "hello"})
	if err == nil || !strings.Contains(err.Error(), "embed query") {
		t.Fatalf("expected embed query error, got %v", err)
	}
}

func TestRetrieveWithOptionsReturnsSearchError(t *testing.T) {
	searcher := &fakeVectorSearcher{err: errors.New("qdrant down")}
	retriever := newRetrieverWithAdapters(
		&fakeQueryEmbedder{vector: []float64{0.1, 0.2}},
		nil,
		searcher,
		"chunks",
		DefaultRetrieverConfig(),
	)

	_, err := retriever.RetrieveWithOptions(context.Background(), RetrieveRequest{Query: "hello"})
	if err == nil || !strings.Contains(err.Error(), "search qdrant") {
		t.Fatalf("expected search qdrant error, got %v", err)
	}
	if searcher.got == nil {
		t.Fatal("expected search request")
	}
	if searcher.got.CollectionName != "chunks" {
		t.Fatalf("collection = %q, want chunks", searcher.got.CollectionName)
	}
	if len(searcher.got.Vector) != 2 {
		t.Fatalf("vector length = %d, want 2", len(searcher.got.Vector))
	}
}

func TestRetrieveWithOptionsFallsBackToVectorOrderWhenRerankFails(t *testing.T) {
	reranker := &fakeChunkReranker{err: errors.New("rerank down")}
	retriever := newRetrieverWithAdapters(
		&fakeQueryEmbedder{vector: []float64{0.1, 0.2}},
		reranker,
		&fakeVectorSearcher{response: &pb.SearchResponse{Result: []*pb.ScoredPoint{
			scoredPoint(0.8, map[string]*pb.Value{
				qdrantclient.FieldText: stringPayload("first"),
			}),
			scoredPoint(0.7, map[string]*pb.Value{
				qdrantclient.FieldText: stringPayload("second"),
			}),
		}}},
		"chunks",
		RetrieverConfig{SearchTopK: 10, TopK: 2},
	)

	chunks, err := retriever.RetrieveWithOptions(context.Background(), RetrieveRequest{Query: "hello"})
	if err != nil {
		t.Fatalf("RetrieveWithOptions: %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("chunks = %#v, want 2", chunks)
	}
	if chunks[0].Text != "first" || chunks[1].Text != "second" {
		t.Fatalf("expected vector order fallback, got %#v", chunks)
	}
	if got := strings.Join(reranker.docs, "|"); got != "first|second" {
		t.Fatalf("rerank docs = %q, want first|second", got)
	}
}

func TestRetrieveWithOptionsMapsPayloadMetadata(t *testing.T) {
	retriever := newRetrieverWithAdapters(
		&fakeQueryEmbedder{vector: []float64{0.1, 0.2}},
		nil,
		&fakeVectorSearcher{response: &pb.SearchResponse{Result: []*pb.ScoredPoint{
			scoredPoint(0.91, map[string]*pb.Value{
				qdrantclient.FieldText:          stringPayload("chunk text"),
				qdrantclient.FieldSourceID:      stringPayload("source-1"),
				qdrantclient.FieldDocumentID:    stringPayload("doc-1"),
				qdrantclient.FieldChunkID:       stringPayload("chunk-1"),
				qdrantclient.FieldContentHash:   stringPayload("hash-1"),
				qdrantclient.FieldTaskID:        stringPayload("task-1"),
				qdrantclient.FieldSessionID:     stringPayload("session-1"),
				qdrantclient.FieldChunkIdx:      intPayload(7),
				qdrantclient.FieldSourceName:    stringPayload("guide.md"),
				qdrantclient.FieldSourceURL:     stringPayload("https://example.com/guide"),
				qdrantclient.FieldContentType:   stringPayload("markdown"),
				qdrantclient.FieldDocumentTitle: stringPayload("Guide"),
				qdrantclient.FieldHeadingPath:   stringPayload("Guide > Install"),
			}),
		}}},
		"chunks",
		DefaultRetrieverConfig(),
	)

	chunks, err := retriever.RetrieveWithOptions(context.Background(), RetrieveRequest{Query: "hello"})
	if err != nil {
		t.Fatalf("RetrieveWithOptions: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("chunks = %#v, want 1", chunks)
	}
	got := chunks[0]
	if got.Text != "chunk text" || got.TaskID != "task-1" || got.SessionID != "session-1" {
		t.Fatalf("metadata missing: %#v", got)
	}
	if got.SourceID != "source-1" || got.DocumentID != "doc-1" || got.ChunkID != "chunk-1" || got.ContentHash != "hash-1" {
		t.Fatalf("identity metadata missing: %#v", got)
	}
	if got.ChunkIdx != 7 {
		t.Fatalf("ChunkIdx = %d, want 7", got.ChunkIdx)
	}
	if got.SourceName != "guide.md" || got.SourceURL != "https://example.com/guide" {
		t.Fatalf("source metadata missing: %#v", got)
	}
	if got.ContentType != "markdown" || got.DocumentTitle != "Guide" || got.HeadingPath != "Guide > Install" {
		t.Fatalf("document metadata missing: %#v", got)
	}
}

func TestNormalizeRetrieverConfigAppliesDefaultsAndClamp(t *testing.T) {
	got := normalizeRetrieverConfig(RetrieverConfig{
		SearchTopK: -1,
		TopK:       100,
		MinScore:   -0.5,
	})

	if got.SearchTopK != 20 {
		t.Fatalf("SearchTopK = %d, want 20", got.SearchTopK)
	}
	if got.TopK != 20 {
		t.Fatalf("TopK = %d, want clamp to 20", got.TopK)
	}
	if got.MinScore != 0 {
		t.Fatalf("MinScore = %f, want 0", got.MinScore)
	}
}

func TestKeepScoreHonorsDisabledAndEnabledThresholds(t *testing.T) {
	if !keepScore(-0.1, 0) {
		t.Fatal("expected disabled threshold to keep any score")
	}
	if keepScore(0.29, 0.3) {
		t.Fatal("expected low score to be filtered")
	}
	if !keepScore(0.3, 0.3) {
		t.Fatal("expected threshold score to be kept")
	}
}

func TestRelevantChunkFromRerankFallsBackToOriginalText(t *testing.T) {
	hits := []retrievalHit{{
		text:          "original text",
		score:         0.42,
		sourceID:      "source-1",
		documentID:    "doc-1",
		chunkID:       "chunk-1",
		contentHash:   "hash-1",
		taskID:        "task-1",
		sourceURL:     "https://example.com/video",
		sourceName:    "guide.md",
		documentTitle: "Guide",
		headingPath:   "Guide > Install",
	}}

	got, ok := relevantChunkFromRerank(hits, model.RerankResult{Index: 0, Score: 0.9})
	if !ok {
		t.Fatal("expected valid rerank result")
	}
	if got.Text != "original text" {
		t.Fatalf("Text = %q, want original text", got.Text)
	}
	if got.Score != 0.9 {
		t.Fatalf("Score = %f, want 0.9", got.Score)
	}
	if got.HeadingPath != "Guide > Install" {
		t.Fatalf("HeadingPath = %q, want Guide > Install", got.HeadingPath)
	}
	if got.TaskID != "task-1" {
		t.Fatalf("TaskID = %q, want task-1", got.TaskID)
	}
	if got.SourceID != "source-1" || got.DocumentID != "doc-1" || got.ChunkID != "chunk-1" || got.ContentHash != "hash-1" {
		t.Fatalf("identity metadata missing: %#v", got)
	}
	if got.SourceURL != "https://example.com/video" {
		t.Fatalf("SourceURL = %q, want source URL", got.SourceURL)
	}
}

func TestRelevantChunkFromRerankRejectsInvalidIndex(t *testing.T) {
	if _, ok := relevantChunkFromRerank([]retrievalHit{{text: "one"}}, model.RerankResult{Index: 3}); ok {
		t.Fatal("expected invalid index to be rejected")
	}
	if _, ok := relevantChunkFromRerank([]retrievalHit{{text: "one"}}, model.RerankResult{Index: -1}); ok {
		t.Fatal("expected negative index to be rejected")
	}
}

type fakeQueryEmbedder struct {
	vector []float64
	err    error
}

func (f *fakeQueryEmbedder) EmbedSingle(context.Context, string) ([]float64, error) {
	if f.err != nil {
		return nil, f.err
	}
	return f.vector, nil
}

type fakeVectorSearcher struct {
	response *pb.SearchResponse
	err      error
	got      *pb.SearchPoints
}

func (f *fakeVectorSearcher) Search(_ context.Context, req *pb.SearchPoints) (*pb.SearchResponse, error) {
	f.got = req
	if f.err != nil {
		return nil, f.err
	}
	if f.response == nil {
		return &pb.SearchResponse{}, nil
	}
	return f.response, nil
}

type fakeChunkReranker struct {
	results []model.RerankResult
	err     error
	docs    []string
}

func (f *fakeChunkReranker) Rerank(_ context.Context, _ string, docs []string) ([]model.RerankResult, error) {
	f.docs = append([]string(nil), docs...)
	if f.err != nil {
		return nil, f.err
	}
	return f.results, nil
}

func scoredPoint(score float32, payload map[string]*pb.Value) *pb.ScoredPoint {
	return &pb.ScoredPoint{
		Score:   score,
		Payload: payload,
	}
}

func stringPayload(value string) *pb.Value {
	return &pb.Value{Kind: &pb.Value_StringValue{StringValue: value}}
}

func intPayload(value int64) *pb.Value {
	return &pb.Value{Kind: &pb.Value_IntegerValue{IntegerValue: value}}
}
