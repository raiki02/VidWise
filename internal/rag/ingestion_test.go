package rag

import (
	"strings"
	"testing"

	qdrantclient "github.com/raiki02/vidwise/internal/storage/qdrant"
)

func TestDetectContentFormat(t *testing.T) {
	tests := []struct {
		name        string
		format      ContentFormat
		filename    string
		contentType string
		want        ContentFormat
	}{
		{
			name:     "explicit markdown wins",
			format:   ContentFormatMarkdown,
			filename: "notes.txt",
			want:     ContentFormatMarkdown,
		},
		{
			name:     "markdown extension",
			filename: "notes.md",
			want:     ContentFormatMarkdown,
		},
		{
			name:        "markdown mime with charset",
			filename:    "notes.txt",
			contentType: "text/markdown; charset=utf-8",
			want:        ContentFormatMarkdown,
		},
		{
			name:        "plain fallback",
			filename:    "notes.txt",
			contentType: "text/plain",
			want:        ContentFormatPlain,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := DetectContentFormat(tt.format, tt.filename, tt.contentType); got != tt.want {
				t.Fatalf("DetectContentFormat() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestDocumentsFromSourceParsesMarkdownWithSourceMetadata(t *testing.T) {
	docs, format := DocumentsFromSource(Source{
		Text: strings.Join([]string{
			"# Guide",
			"",
			"## Upload",
			"",
			"Upload markdown files.",
		}, "\n"),
		Filename:    "guide.md",
		ContentType: "text/markdown",
	})

	if format != ContentFormatMarkdown {
		t.Fatalf("format = %q, want %q", format, ContentFormatMarkdown)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 markdown section, got %#v", docs)
	}

	doc := docs[0]
	if got := doc.Metadata[qdrantclient.FieldContentType]; got != MarkdownContentType {
		t.Fatalf("content type metadata = %q, want %q", got, MarkdownContentType)
	}
	if got := doc.Metadata[qdrantclient.FieldSourceName]; got != "guide.md" {
		t.Fatalf("source name metadata = %q, want guide.md", got)
	}
	if got := doc.Metadata[qdrantclient.FieldHeadingPath]; got != "Guide > Upload" {
		t.Fatalf("heading path metadata = %q, want Guide > Upload", got)
	}
	if !strings.Contains(doc.PageContent, "# Guide\n\n## Upload\n\nUpload markdown files.") {
		t.Fatalf("expected heading context merged into content, got %q", doc.PageContent)
	}
}

func TestDocumentsFromSourceKeepsPlainTextAsSingleDocument(t *testing.T) {
	docs, format := DocumentsFromSource(Source{
		Text:     "first paragraph\n\nsecond paragraph",
		Filename: "notes.txt",
		Metadata: map[string]string{
			"tenant": "demo",
		},
	})

	if format != ContentFormatPlain {
		t.Fatalf("format = %q, want %q", format, ContentFormatPlain)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 document, got %#v", docs)
	}

	doc := docs[0]
	if got := doc.Metadata[qdrantclient.FieldContentType]; got != PlainContentType {
		t.Fatalf("content type metadata = %q, want %q", got, PlainContentType)
	}
	if got := doc.Metadata[qdrantclient.FieldSourceName]; got != "notes.txt" {
		t.Fatalf("source name metadata = %q, want notes.txt", got)
	}
	if got := doc.Metadata["tenant"]; got != "demo" {
		t.Fatalf("custom metadata = %q, want demo", got)
	}
}

func TestIndexOptionsDoNotMutateIndexerDefaults(t *testing.T) {
	idx := NewIndexer(nil, nil, "test")
	idx.SetChunkParams(2048, 256)
	before := idx.defaultChunkConfig()

	overlap := 0
	custom := idx.chunkConfigWithOptions(IndexOptions{
		ChunkRunes:   64,
		OverlapRunes: &overlap,
	})
	after := idx.defaultChunkConfig()

	if after != before {
		t.Fatalf("per-call options mutated indexer defaults: before=%#v after=%#v", before, after)
	}
	if custom.MaxRunes != 64 {
		t.Fatalf("custom MaxRunes = %d, want 64", custom.MaxRunes)
	}
	if custom.OverlapRunes != 0 {
		t.Fatalf("custom OverlapRunes = %d, want 0", custom.OverlapRunes)
	}
}
