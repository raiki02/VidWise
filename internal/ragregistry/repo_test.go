package ragregistry

import (
	"testing"

	"github.com/raiki02/vidwise/internal/rag"
)

func TestSourceRecordFromSummaryNormalizesMetadata(t *testing.T) {
	got, ok := sourceRecordFromSummary(rag.SourceSummary{
		SourceID:      " source-1 ",
		UserID:        " user-1 ",
		SessionID:     " session-1 ",
		SourceName:    " guide.md ",
		SourceURL:     " https://example.com/guide ",
		ContentType:   " markdown ",
		DocumentTitle: " Guide ",
		ChunkCount:    3,
	})
	if !ok {
		t.Fatal("expected record")
	}

	if got.SourceID != "source-1" || got.UserID != "user-1" || got.SessionID != "session-1" {
		t.Fatalf("scope/source fields not normalized: %#v", got)
	}
	if got.SourceName != "guide.md" || got.SourceURL != "https://example.com/guide" || got.DocumentTitle != "Guide" {
		t.Fatalf("source metadata not normalized: %#v", got)
	}
	if got.ContentType != "markdown" || got.ChunkCount != 3 || got.Status != StatusActive || got.DeletedAt != nil {
		t.Fatalf("lifecycle metadata unexpected: %#v", got)
	}
}

func TestSourceRecordFromSummaryRejectsMissingSourceID(t *testing.T) {
	if _, ok := sourceRecordFromSummary(rag.SourceSummary{SourceName: "guide.md"}); ok {
		t.Fatal("expected missing source id to be rejected")
	}
}

func TestSourceSummaryFromRecordKeepsOperationalFields(t *testing.T) {
	got := sourceSummaryFromRecord(SourceRecord{
		SourceID:      "source-1",
		UserID:        "user-1",
		SessionID:     "session-1",
		SourceName:    "guide.md",
		SourceURL:     "https://example.com/guide",
		ContentType:   "markdown",
		DocumentTitle: "Guide",
		ChunkCount:    2,
	})

	if got.SourceID != "source-1" || got.UserID != "user-1" || got.SessionID != "session-1" {
		t.Fatalf("identity metadata missing: %#v", got)
	}
	if got.SourceName != "guide.md" || got.SourceURL != "https://example.com/guide" || got.ContentType != "markdown" {
		t.Fatalf("source metadata missing: %#v", got)
	}
	if got.DocumentTitle != "Guide" || got.ChunkCount != 2 {
		t.Fatalf("summary metadata missing: %#v", got)
	}
}

func TestNormalizeSourceIDsTrimsAndDeduplicates(t *testing.T) {
	got := normalizeSourceIDs([]string{" source-1 ", "", "source-2", "source-1"})
	if len(got) != 2 || got[0] != "source-1" || got[1] != "source-2" {
		t.Fatalf("unexpected source ids: %#v", got)
	}
}
