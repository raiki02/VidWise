package rag

import (
	"strings"
	"testing"

	qdrantclient "github.com/raiki02/vidwise/internal/storage/qdrant"
)

func TestParseMarkdownDocumentsBuildsHeadingMetadataAndMergesParagraphs(t *testing.T) {
	input := strings.Join([]string{
		"# Guide",
		"",
		"Opening paragraph.",
		"",
		"## Install",
		"",
		"First paragraph.",
		"",
		"Second paragraph.",
		"",
		"### CLI",
		"",
		"- make deps",
		"- make run",
		"",
		"## Usage",
		"",
		"Ask a question.",
	}, "\n")

	docs := ParseMarkdownDocuments(input, map[string]string{
		qdrantclient.FieldSourceName: "guide.md",
	})

	if len(docs) != 4 {
		t.Fatalf("expected 4 heading documents, got %#v", docs)
	}

	install := docs[1]
	if got := install.Metadata[qdrantclient.FieldHeadingPath]; got != "Guide > Install" {
		t.Fatalf("unexpected heading path: %q", got)
	}
	if got := install.Metadata[qdrantclient.FieldHeader1]; got != "Guide" {
		t.Fatalf("unexpected h1 metadata: %q", got)
	}
	if got := install.Metadata[qdrantclient.FieldHeader2]; got != "Install" {
		t.Fatalf("unexpected h2 metadata: %q", got)
	}
	if got := install.Metadata[qdrantclient.FieldHeadingLevel]; got != "2" {
		t.Fatalf("unexpected heading level: %q", got)
	}
	if got := install.Metadata[qdrantclient.FieldSourceName]; got != "guide.md" {
		t.Fatalf("source metadata not preserved: %q", got)
	}
	if !strings.Contains(install.PageContent, "# Guide\n\n## Install") {
		t.Fatalf("expected parent and current headings in content, got %q", install.PageContent)
	}
	if !strings.Contains(install.PageContent, "First paragraph.\n\nSecond paragraph.") {
		t.Fatalf("expected section paragraphs to be merged, got %q", install.PageContent)
	}

	cli := docs[2]
	if got := cli.Metadata[qdrantclient.FieldHeadingPath]; got != "Guide > Install > CLI" {
		t.Fatalf("unexpected nested heading path: %q", got)
	}
	if !strings.Contains(cli.PageContent, "- make deps\n- make run") {
		t.Fatalf("expected list content to be preserved, got %q", cli.PageContent)
	}
}

func TestParseMarkdownDocumentsPreservesFencedCodeBlocks(t *testing.T) {
	input := strings.Join([]string{
		"# Notes",
		"",
		"```go",
		`fmt.Println("hello")`,
		"```",
	}, "\n")

	docs := ParseMarkdownDocuments(input, nil)
	if len(docs) != 1 {
		t.Fatalf("expected 1 document, got %#v", docs)
	}
	if !strings.Contains(docs[0].PageContent, "```go\nfmt.Println(\"hello\")\n```") {
		t.Fatalf("expected fenced code block to be preserved, got %q", docs[0].PageContent)
	}
}

func TestParseMarkdownDocumentsPreservesInlineSpacing(t *testing.T) {
	input := "# Notes\n\nBefore [linked text](https://example.com) after **bold text**."

	docs := ParseMarkdownDocuments(input, nil)
	if len(docs) != 1 {
		t.Fatalf("expected 1 document, got %#v", docs)
	}
	if !strings.Contains(docs[0].PageContent, "Before linked text after bold text.") {
		t.Fatalf("expected inline spacing to be preserved, got %q", docs[0].PageContent)
	}
}

func TestParseMarkdownDocumentsSkipsHeadingOnlyParentSections(t *testing.T) {
	input := "# Guide\n\n## Install\n\nSteps."

	docs := ParseMarkdownDocuments(input, nil)
	if len(docs) != 1 {
		t.Fatalf("expected only the child section with body content, got %#v", docs)
	}
	if got := docs[0].Metadata[qdrantclient.FieldHeadingPath]; got != "Guide > Install" {
		t.Fatalf("unexpected heading path: %q", got)
	}
	if !strings.Contains(docs[0].PageContent, "# Guide\n\n## Install\n\nSteps.") {
		t.Fatalf("expected parent heading to remain as child context, got %q", docs[0].PageContent)
	}
}

func TestChunkDocumentsCarriesDocumentMetadata(t *testing.T) {
	input := strings.Join([]string{
		"# Guide",
		"",
		"## Install",
		"",
		"First paragraph. Second paragraph.",
	}, "\n")
	docs := ParseMarkdownDocuments(input, map[string]string{
		qdrantclient.FieldSourceName: "guide.md",
	})

	idx := NewIndexer(nil, nil, "test")
	idx.SetChunkParams(256, 0)
	chunks := idx.chunkDocuments(docs)
	if len(chunks) == 0 {
		t.Fatal("expected chunks")
	}

	got := chunks[len(chunks)-1].Metadata
	if got[qdrantclient.FieldSourceName] != "guide.md" {
		t.Fatalf("source metadata missing: %#v", got)
	}
	if got[qdrantclient.FieldContentType] != MarkdownContentType {
		t.Fatalf("content type metadata missing: %#v", got)
	}
	if got[qdrantclient.FieldHeadingPath] != "Guide > Install" {
		t.Fatalf("heading path metadata missing: %#v", got)
	}
	if got[qdrantclient.FieldSectionIndex] == "" {
		t.Fatalf("section index metadata missing: %#v", got)
	}
	if got[qdrantclient.FieldSectionChunkIndex] == "" {
		t.Fatalf("section chunk index metadata missing: %#v", got)
	}
	if got[qdrantclient.FieldChunkSource] == "" {
		t.Fatalf("chunk source metadata missing: %#v", got)
	}
	if got[qdrantclient.FieldChunkSource] != string(ChunkSourceSection) {
		t.Fatalf("markdown chunk source = %q, want section", got[qdrantclient.FieldChunkSource])
	}
}

func TestChunkDocumentsUsesPlainChunkingForPlainText(t *testing.T) {
	docs := []Document{{
		PageContent: "# Plain transcript marker\n\nThis is still plain text, not markdown.",
		Metadata: map[string]string{
			qdrantclient.FieldContentType: PlainContentType,
			qdrantclient.FieldSourceName:  "notes.txt",
		},
	}}

	idx := NewIndexer(nil, nil, "test")
	idx.SetChunkParams(512, 0)
	chunks := idx.chunkDocuments(docs)
	if len(chunks) != 1 {
		t.Fatalf("expected one packed chunk, got %#v", chunks)
	}
	if got := chunks[0].Metadata[qdrantclient.FieldChunkSource]; got != string(ChunkSourceParagraph) {
		t.Fatalf("plain chunk source = %q, want paragraph", got)
	}
}

func TestChunkIdentityIsStableForSameSourceAndPosition(t *testing.T) {
	base := documentChunk{
		Text: "first version",
		Metadata: map[string]string{
			qdrantclient.FieldSourceName:        "guide.md",
			qdrantclient.FieldContentType:       MarkdownContentType,
			qdrantclient.FieldDocumentTitle:     "Guide",
			qdrantclient.FieldHeadingPath:       "Guide > Install",
			qdrantclient.FieldSectionIndex:      "1",
			qdrantclient.FieldSectionChunkIndex: "0",
			qdrantclient.FieldChunkSource:       string(ChunkSourceSection),
		},
	}
	changed := base
	changed.Text = "second version"

	first := newChunkIdentity(base, "user-1", "session-1", 0)
	second := newChunkIdentity(changed, "user-1", "session-1", 0)

	if first.sourceID != second.sourceID {
		t.Fatalf("sourceID changed for same source: %q != %q", first.sourceID, second.sourceID)
	}
	if first.documentID != second.documentID {
		t.Fatalf("documentID changed for same source: %q != %q", first.documentID, second.documentID)
	}
	if first.chunkID != second.chunkID {
		t.Fatalf("chunkID changed for same source position: %q != %q", first.chunkID, second.chunkID)
	}
	if first.contentHash == second.contentHash {
		t.Fatalf("content hash did not change for changed text: %q", first.contentHash)
	}
	if stablePointUUID(first.chunkID) != stablePointUUID(second.chunkID) {
		t.Fatalf("stable point UUID changed for same chunk id")
	}
}

func TestChunkIdentitySharesSourceAcrossMarkdownSections(t *testing.T) {
	base := documentChunk{
		Text: "install section",
		Metadata: map[string]string{
			qdrantclient.FieldSourceName:        "guide.md",
			qdrantclient.FieldContentType:       MarkdownContentType,
			qdrantclient.FieldDocumentTitle:     "Guide",
			qdrantclient.FieldHeadingPath:       "Guide > Install",
			qdrantclient.FieldSectionIndex:      "0",
			qdrantclient.FieldSectionChunkIndex: "0",
			qdrantclient.FieldChunkSource:       string(ChunkSourceSection),
		},
	}
	otherSection := documentChunk{
		Text: "usage section",
		Metadata: mergeMetadata(base.Metadata, map[string]string{
			qdrantclient.FieldHeadingPath:       "Guide > Usage",
			qdrantclient.FieldSectionIndex:      "1",
			qdrantclient.FieldSectionChunkIndex: "0",
		}),
	}

	first := newChunkIdentity(base, "user-1", "session-1", 0)
	second := newChunkIdentity(otherSection, "user-1", "session-1", 1)

	if first.sourceID != second.sourceID {
		t.Fatalf("sections from the same file should share sourceID: %q != %q", first.sourceID, second.sourceID)
	}
	if first.documentID == second.documentID {
		t.Fatalf("different markdown sections should keep distinct documentID")
	}
	if first.chunkID == second.chunkID {
		t.Fatalf("different markdown chunks should keep distinct chunkID")
	}
}

func TestSourceIDsFromIdentitiesReturnsUniqueStableOrder(t *testing.T) {
	got := sourceIDsFromIdentities([]chunkIdentity{
		{sourceID: "source-1"},
		{sourceID: "source-2"},
		{sourceID: "source-1"},
		{sourceID: " "},
	})

	if len(got) != 2 {
		t.Fatalf("expected two unique source ids, got %#v", got)
	}
	if got[0] != "source-1" || got[1] != "source-2" {
		t.Fatalf("source ids should preserve first-seen order, got %#v", got)
	}
}

func TestSourceSummariesFromChunksAggregatesSourceMetadata(t *testing.T) {
	chunks := []documentChunk{
		{
			Text: "install",
			Metadata: map[string]string{
				qdrantclient.FieldSourceName:    "guide.md",
				qdrantclient.FieldSourceURL:     "https://example.com/guide",
				qdrantclient.FieldContentType:   MarkdownContentType,
				qdrantclient.FieldDocumentTitle: "Guide",
			},
		},
		{
			Text: "usage",
			Metadata: map[string]string{
				qdrantclient.FieldSourceName:  "guide.md",
				qdrantclient.FieldContentType: MarkdownContentType,
			},
		},
	}

	got := sourceSummariesFromChunks(chunks, []chunkIdentity{
		{sourceID: "source-1"},
		{sourceID: "source-1"},
	}, " user-1 ", " session-1 ")

	if len(got) != 1 {
		t.Fatalf("expected one source summary, got %#v", got)
	}
	if got[0].SourceID != "source-1" || got[0].ChunkCount != 2 {
		t.Fatalf("unexpected source identity/count: %#v", got[0])
	}
	if got[0].UserID != "user-1" || got[0].SessionID != "session-1" {
		t.Fatalf("scope metadata missing: %#v", got[0])
	}
	if got[0].SourceName != "guide.md" || got[0].SourceURL != "https://example.com/guide" || got[0].DocumentTitle != "Guide" {
		t.Fatalf("source metadata missing: %#v", got[0])
	}
}

func TestNormalizeSourceIDsTrimsAndDeduplicates(t *testing.T) {
	got := normalizeSourceIDs([]string{" source-1 ", "", "source-2", "source-1"})

	if len(got) != 2 || got[0] != "source-1" || got[1] != "source-2" {
		t.Fatalf("unexpected source ids: %#v", got)
	}
}

func TestDeleteSourcesRejectsMissingSourceID(t *testing.T) {
	idx := NewIndexer(nil, nil, "test")

	if _, err := idx.DeleteSources(nil, []string{" ", ""}); err == nil {
		t.Fatal("expected source_id validation error")
	}
}

func TestDeleteSourcesRejectsMissingQdrantClient(t *testing.T) {
	idx := NewIndexer(nil, nil, "test")

	if _, err := idx.DeleteSource(nil, "source-1"); err == nil {
		t.Fatal("expected qdrant client error")
	}
}

func TestChunkIdentityScopesByTenantAndSource(t *testing.T) {
	chunk := documentChunk{
		Text: "same content",
		Metadata: map[string]string{
			qdrantclient.FieldSourceName:        "guide.md",
			qdrantclient.FieldContentType:       MarkdownContentType,
			qdrantclient.FieldSectionChunkIndex: "0",
			qdrantclient.FieldChunkSource:       string(ChunkSourceSection),
		},
	}

	userOne := newChunkIdentity(chunk, "user-1", "", 0)
	userTwo := newChunkIdentity(chunk, "user-2", "", 0)
	if userOne.sourceID == userTwo.sourceID {
		t.Fatalf("sourceID should be scoped by user")
	}
	if userOne.chunkID == userTwo.chunkID {
		t.Fatalf("chunkID should be scoped by user")
	}

	otherSource := chunk
	otherSource.Metadata = mergeMetadata(chunk.Metadata, map[string]string{
		qdrantclient.FieldSourceName: "other.md",
	})
	fromOtherSource := newChunkIdentity(otherSource, "user-1", "", 0)
	if userOne.sourceID == fromOtherSource.sourceID {
		t.Fatalf("sourceID should differ across sources")
	}
	if userOne.chunkID == fromOtherSource.chunkID {
		t.Fatalf("chunkID should differ across sources")
	}
}

func TestChunkIdentityFallsBackToContentForAnonymousText(t *testing.T) {
	chunk := documentChunk{Text: "anonymous content", Metadata: map[string]string{}}
	again := documentChunk{Text: "anonymous content", Metadata: map[string]string{}}
	changed := documentChunk{Text: "changed anonymous content", Metadata: map[string]string{}}

	first := newChunkIdentity(chunk, "", "", 0)
	second := newChunkIdentity(again, "", "", 0)
	third := newChunkIdentity(changed, "", "", 0)

	if first.chunkID != second.chunkID {
		t.Fatalf("anonymous identical content should have stable chunkID")
	}
	if first.chunkID == third.chunkID {
		t.Fatalf("anonymous changed content should produce a new chunkID")
	}
}

func TestBuildSourceDeleteRequestFiltersBySourceID(t *testing.T) {
	req := buildSourceDeleteRequest("rag_docs", "source-123", &RetrieveFilter{
		UserID:    "user-1",
		SessionID: "session-1",
	})

	if req.CollectionName != "rag_docs" {
		t.Fatalf("unexpected collection: %q", req.CollectionName)
	}
	if !req.GetWait() {
		t.Fatalf("delete request should wait for completion")
	}

	filter := req.GetPoints().GetFilter()
	if filter == nil || len(filter.GetMust()) != 3 {
		t.Fatalf("expected source and scope filters, got %#v", filter)
	}
	got := map[string]string{}
	for _, cond := range filter.GetMust() {
		field := cond.GetField()
		if field == nil {
			t.Fatalf("expected field condition")
		}
		got[field.GetKey()] = field.GetMatch().GetKeyword()
	}
	if got[qdrantclient.FieldSourceID] != "source-123" {
		t.Fatalf("source filter missing: %#v", got)
	}
	if got[qdrantclient.FieldUserID] != "user-1" {
		t.Fatalf("user filter missing: %#v", got)
	}
	if got[qdrantclient.FieldSessionID] != "session-1" {
		t.Fatalf("session filter missing: %#v", got)
	}
}
