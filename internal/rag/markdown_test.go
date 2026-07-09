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
}
