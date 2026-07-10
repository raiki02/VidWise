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
		DocumentIDs:   []string{" doc-1 ", "doc-1", "doc-2"},
		TaskIDs:       []string{" task-1 "},
		HeadingPaths:  []string{" Guide > Install "},
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
	if got.DocumentIDs != `["doc-1","doc-2"]` || got.TaskIDs != `["task-1"]` || got.HeadingPaths != `["Guide \u003e Install"]` {
		t.Fatalf("structured metadata not encoded: %#v", got)
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
		DocumentIDs:   `["doc-1","doc-2"]`,
		TaskIDs:       `["task-1"]`,
		HeadingPaths:  `["Guide > Install"]`,
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
	if len(got.DocumentIDs) != 2 || got.DocumentIDs[0] != "doc-1" || got.DocumentIDs[1] != "doc-2" {
		t.Fatalf("document ids missing: %#v", got)
	}
	if len(got.TaskIDs) != 1 || got.TaskIDs[0] != "task-1" {
		t.Fatalf("task ids missing: %#v", got)
	}
	if len(got.HeadingPaths) != 1 || got.HeadingPaths[0] != "Guide > Install" {
		t.Fatalf("heading paths missing: %#v", got)
	}
}

func TestNormalizeSourceIDsTrimsAndDeduplicates(t *testing.T) {
	got := normalizeSourceIDs([]string{" source-1 ", "", "source-2", "source-1"})
	if len(got) != 2 || got[0] != "source-1" || got[1] != "source-2" {
		t.Fatalf("unexpected source ids: %#v", got)
	}
}
