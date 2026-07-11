package tool

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/raiki02/vidwise/internal/rag"
)

func TestNewRAGIndexToolRejectsNilManager(t *testing.T) {
	inner, wrapper, err := NewRAGIndexTool(nil)
	if err == nil {
		t.Fatal("expected error for nil source manager")
	}
	if inner != nil || wrapper != nil {
		t.Fatalf("expected no tool on error, got inner=%v wrapper=%v", inner, wrapper)
	}
	if !strings.Contains(err.Error(), "source manager") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestNewRAGQueryToolRejectsNilRetriever(t *testing.T) {
	inner, wrapper, err := NewRAGQueryTool(nil)
	if err == nil {
		t.Fatal("expected error for nil retriever")
	}
	if inner != nil || wrapper != nil {
		t.Fatalf("expected no tool on error, got inner=%v wrapper=%v", inner, wrapper)
	}
	if !strings.Contains(err.Error(), "rag retriever") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestNewRAGDeleteToolRejectsNilManager(t *testing.T) {
	inner, wrapper, err := NewRAGDeleteTool(nil)
	if err == nil {
		t.Fatal("expected error for nil source manager")
	}
	if inner != nil || wrapper != nil {
		t.Fatalf("expected no tool on error, got inner=%v wrapper=%v", inner, wrapper)
	}
	if !strings.Contains(err.Error(), "source manager") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestNewRAGListSourcesToolRejectsNilManager(t *testing.T) {
	inner, wrapper, err := NewRAGListSourcesTool(nil)
	if err == nil {
		t.Fatal("expected error for nil source manager")
	}
	if inner != nil || wrapper != nil {
		t.Fatalf("expected no tool on error, got inner=%v wrapper=%v", inner, wrapper)
	}
	if !strings.Contains(err.Error(), "source manager") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestNormalizeRAGIndexInputRequiresScope(t *testing.T) {
	if _, err := normalizeRAGIndexInput(RAGIndexInput{Text: "hello world"}); err == nil {
		t.Fatal("expected missing scope error")
	}
}

func TestNormalizeRAGIndexInputKeepsPlainTextInputWithScope(t *testing.T) {
	req, err := normalizeRAGIndexInput(RAGIndexInput{
		Text:      "  hello world  ",
		SessionID: " session-1 ",
	})
	if err != nil {
		t.Fatalf("normalize input: %v", err)
	}

	if req.Source.Text != "hello world" {
		t.Fatalf("Text = %q, want hello world", req.Source.Text)
	}
	if req.Source.Format != rag.ContentFormatAuto {
		t.Fatalf("Format = %q, want auto", req.Source.Format)
	}
	if req.UserID != "" || req.SessionID != "session-1" {
		t.Fatalf("expected session-scoped input, got user=%q session=%q", req.UserID, req.SessionID)
	}
}

func TestNormalizeRAGIndexInputBuildsStructuredMarkdownSource(t *testing.T) {
	overlap := 0
	req, err := normalizeRAGIndexInput(RAGIndexInput{
		Text:        "# Guide\n\nBody.",
		Filename:    " guide.md ",
		ContentType: " text/markdown ",
		Format:      "markdown",
		Metadata: map[string]string{
			"source_url": "https://example.com/video",
		},
		UserID:       " u1 ",
		SessionID:    " s1 ",
		ChunkRunes:   256,
		OverlapRunes: &overlap,
	})
	if err != nil {
		t.Fatalf("normalize input: %v", err)
	}

	if req.Source.Format != rag.ContentFormatMarkdown {
		t.Fatalf("Format = %q, want markdown", req.Source.Format)
	}
	if req.Source.Filename != "guide.md" {
		t.Fatalf("Filename = %q, want guide.md", req.Source.Filename)
	}
	if req.Source.ContentType != "text/markdown" {
		t.Fatalf("ContentType = %q, want text/markdown", req.Source.ContentType)
	}
	if req.Source.Metadata["source_url"] != "https://example.com/video" {
		t.Fatalf("metadata missing: %#v", req.Source.Metadata)
	}
	if req.UserID != "u1" || req.SessionID != "" {
		t.Fatalf("scope not normalized: user=%q session=%q", req.UserID, req.SessionID)
	}
	if req.Options.ChunkRunes != 256 {
		t.Fatalf("ChunkRunes = %d, want 256", req.Options.ChunkRunes)
	}
	if req.Options.OverlapRunes == nil || *req.Options.OverlapRunes != 0 {
		t.Fatalf("OverlapRunes = %v, want 0", req.Options.OverlapRunes)
	}
}

func TestNormalizeRAGIndexInputRejectsEmptyTextAndUnknownFormat(t *testing.T) {
	if _, err := normalizeRAGIndexInput(RAGIndexInput{Text: "   "}); err == nil {
		t.Fatal("expected empty text error")
	}

	_, err := normalizeRAGIndexInput(RAGIndexInput{
		Text:   "hello",
		Format: "pdf",
	})
	if err == nil {
		t.Fatal("expected format error")
	}
	if !strings.Contains(err.Error(), "format") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestNormalizeRAGQueryInputRequiresQueryAndScope(t *testing.T) {
	if _, err := normalizeRAGQueryInput(RAGQueryInput{Query: "   ", UserID: "u1"}); err == nil {
		t.Fatal("expected query error")
	}
	if _, err := normalizeRAGQueryInput(RAGQueryInput{Query: "hello"}); err == nil {
		t.Fatal("expected missing scope error")
	}
}

func TestNormalizeRAGQueryInputBuildsStrictRequest(t *testing.T) {
	minScore := 0.3
	req, err := normalizeRAGQueryInput(RAGQueryInput{
		Query:        " hello ",
		SessionID:    " session-1 ",
		SourceIDs:    []string{" source-1 ", "source-2", "source-1", " "},
		DocumentIDs:  []string{" doc-1 ", "doc-1"},
		TopK:         4,
		SearchTopK:   12,
		MinScore:     &minScore,
		ContextRunes: 90,
	})
	if err != nil {
		t.Fatalf("normalize query input: %v", err)
	}
	if req.Retrieve.Query != "hello" {
		t.Fatalf("query = %q, want hello", req.Retrieve.Query)
	}
	if req.Retrieve.Filter == nil || req.Retrieve.Filter.SessionID != "session-1" {
		t.Fatalf("unexpected filter: %#v", req.Retrieve.Filter)
	}
	if len(req.Retrieve.Filter.SourceIDs) != 2 || req.Retrieve.Filter.SourceIDs[0] != "source-1" || req.Retrieve.Filter.SourceIDs[1] != "source-2" {
		t.Fatalf("unexpected source ids: %#v", req.Retrieve.Filter.SourceIDs)
	}
	if len(req.Retrieve.Filter.DocumentIDs) != 1 || req.Retrieve.Filter.DocumentIDs[0] != "doc-1" {
		t.Fatalf("unexpected document ids: %#v", req.Retrieve.Filter.DocumentIDs)
	}
	if req.Retrieve.TopK != 4 || req.Retrieve.SearchTopK != 12 || req.Retrieve.MinScore == nil || *req.Retrieve.MinScore != minScore {
		t.Fatalf("unexpected retrieval options: %#v", req)
	}
	if req.Context.MaxRunes != 90 {
		t.Fatalf("context max runes = %d, want 90", req.Context.MaxRunes)
	}
}

func TestRAGQueryOutputReportsRetrievedStatusAndCount(t *testing.T) {
	out := newRAGQueryOutput([]rag.RelevantChunk{
		{Text: "chunk one", Score: 0.91, SourceName: "guide.md", HeadingPath: "Intro"},
		{Text: "chunk two", Score: 0.82, SourceName: "guide.md", HeadingPath: "Details"},
	}, rag.ContextConfig{MaxRunes: 1024})

	if out.Status != "retrieved" {
		t.Fatalf("status = %q, want retrieved", out.Status)
	}
	if out.Count != 2 {
		t.Fatalf("count = %d, want 2", out.Count)
	}
	if out.ContextUsedChunks != 2 || out.ContextTruncated {
		t.Fatalf("unexpected context outcome: %#v", out)
	}
	if len(out.ContextCitations) != 2 || out.ContextCitations[0].SnippetNumber != 1 || out.ContextCitations[0].Chunk.SourceName != "guide.md" {
		t.Fatalf("unexpected context citations: %#v", out.ContextCitations)
	}
	for _, want := range []string{"[片段 1]", "guide.md / Intro", "chunk two"} {
		if !strings.Contains(out.Context, want) {
			t.Fatalf("expected context to contain %q, got %q", want, out.Context)
		}
	}
	encoded, err := json.Marshal(out)
	if err != nil {
		t.Fatalf("marshal output: %v", err)
	}
	for _, want := range []string{`"status":"retrieved"`, `"count":2`, `"chunks":[`, `"context":"`, `"context_used_chunks":2`, `"context_citations":[`} {
		if !strings.Contains(string(encoded), want) {
			t.Fatalf("expected %s in output JSON, got %s", want, encoded)
		}
	}
}

func TestRAGQueryOutputReportsNoResults(t *testing.T) {
	out := newRAGQueryOutput(nil, rag.ContextConfig{MaxRunes: 1024})

	if out.Status != "no_results" {
		t.Fatalf("status = %q, want no_results", out.Status)
	}
	if out.Count != 0 {
		t.Fatalf("count = %d, want 0", out.Count)
	}
	if len(out.Chunks) != 0 {
		t.Fatalf("chunks = %#v, want empty", out.Chunks)
	}
	if out.Context != "" || out.ContextUsedChunks != 0 || len(out.ContextCitations) != 0 {
		t.Fatalf("unexpected context for empty result: %#v", out)
	}
}

func TestRAGQueryOutputPacksPromptContextWithBudgetAndDedup(t *testing.T) {
	out := newRAGQueryOutput([]rag.RelevantChunk{
		{Text: "same context", Score: 0.91, ContentHash: "hash-1", SourceName: "guide.md"},
		{Text: "same context", Score: 0.89, ContentHash: "hash-1", SourceName: "guide.md"},
		{Text: strings.Repeat("界", 200), Score: 0.82, ContentHash: "hash-2", SourceName: "long.md"},
	}, rag.ContextConfig{MaxRunes: 120})

	if out.ContextUsedChunks != 2 {
		t.Fatalf("context used chunks = %d, want 2", out.ContextUsedChunks)
	}
	if out.ContextSkippedDuplicates != 1 {
		t.Fatalf("context skipped duplicates = %d, want 1", out.ContextSkippedDuplicates)
	}
	if !out.ContextTruncated {
		t.Fatalf("expected context truncation, got %#v", out)
	}
	if strings.Count(out.Context, "same context") != 1 {
		t.Fatalf("duplicate leaked into packed context: %q", out.Context)
	}
}

func TestNormalizeRAGDeleteInputAcceptsSingleAndBatchSourceIDs(t *testing.T) {
	got, err := normalizeRAGDeleteInput(RAGDeleteInput{
		SourceID:  " source-1 ",
		SourceIDs: []string{"source-2", "source-1", " "},
		UserID:    " user-1 ",
		SessionID: " session-1 ",
	})
	if err != nil {
		t.Fatalf("normalize delete input: %v", err)
	}

	if len(got.SourceIDs) != 2 || got.SourceIDs[0] != "source-1" || got.SourceIDs[1] != "source-2" {
		t.Fatalf("unexpected source ids: %#v", got.SourceIDs)
	}
	if got.Filter == nil || got.Filter.UserID != "user-1" || got.Filter.SessionID != "" {
		t.Fatalf("unexpected filter: %#v", got.Filter)
	}
}

func TestNormalizeRAGDeleteInputRejectsMissingSourceID(t *testing.T) {
	if _, err := normalizeRAGDeleteInput(RAGDeleteInput{}); err == nil {
		t.Fatal("expected missing source_id error")
	}
}

func TestNormalizeRAGDeleteInputRejectsMissingScope(t *testing.T) {
	if _, err := normalizeRAGDeleteInput(RAGDeleteInput{SourceID: "source-1"}); err == nil {
		t.Fatal("expected missing scope error")
	}
}

func TestNormalizeRAGListSourcesInputRequiresScope(t *testing.T) {
	if _, err := normalizeRAGListSourcesInput(RAGListSourcesInput{}); err == nil {
		t.Fatal("expected missing scope error")
	}
}

func TestNormalizeRAGListSourcesInputBuildsStrictRequest(t *testing.T) {
	req, err := normalizeRAGListSourcesInput(RAGListSourcesInput{
		UserID: " u1 ",
		Limit:  20,
	})
	if err != nil {
		t.Fatalf("normalize list sources input: %v", err)
	}
	if req.Filter == nil || req.Filter.UserID != "u1" {
		t.Fatalf("unexpected filter: %#v", req.Filter)
	}
	if req.Limit != 20 {
		t.Fatalf("limit = %d, want 20", req.Limit)
	}
}

func TestNormalizeRAGListSourcesInputRejectsNegativeLimit(t *testing.T) {
	if _, err := normalizeRAGListSourcesInput(RAGListSourcesInput{UserID: "u1", Limit: -1}); err == nil {
		t.Fatal("expected limit error")
	}
}

func TestRAGIndexOutputIncludesSourceIDs(t *testing.T) {
	out := RAGIndexOutput{
		ChunkCount:  2,
		ContentType: "markdown",
		SourceIDs:   []string{"source-1"},
		Sources: []rag.SourceSummary{
			{SourceID: "source-1", SourceName: "guide.md", ChunkCount: 2},
		},
	}

	encoded, err := json.Marshal(out)
	if err != nil {
		t.Fatalf("marshal output: %v", err)
	}
	if !strings.Contains(string(encoded), `"source_ids":["source-1"]`) {
		t.Fatalf("expected source ids in output JSON, got %s", encoded)
	}
	if !strings.Contains(string(encoded), `"sources":[`) {
		t.Fatalf("expected source summaries in output JSON, got %s", encoded)
	}
}
