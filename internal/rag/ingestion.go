package rag

import (
	"path/filepath"
	"strings"

	qdrantclient "github.com/raiki02/vidwise/internal/storage/qdrant"
)

// ContentFormat is the source format used by the ingestion module.
type ContentFormat string

const (
	ContentFormatAuto     ContentFormat = ""
	ContentFormatPlain    ContentFormat = PlainContentType
	ContentFormatMarkdown ContentFormat = MarkdownContentType
)

// Source is a raw text-like input to the RAG ingestion module.
type Source struct {
	Text        string
	Filename    string
	ContentType string
	Format      ContentFormat
	Metadata    map[string]string
}

// IndexOptions controls per-call indexing behaviour without mutating the
// indexer's defaults.
type IndexOptions struct {
	ChunkRunes   int
	OverlapRunes *int
}

// IndexResult describes the indexed source.
type IndexResult struct {
	ChunkCount  int
	ContentType string
	SourceIDs   []string
	Sources     []SourceSummary
}

// DeleteResult describes a source deletion request.
type DeleteResult struct {
	SourceIDs []string
}

// DeleteRequest describes a source deletion request. Filter is optional for
// admin/global deletion, and should be set for tenant/session-scoped callers.
type DeleteRequest struct {
	SourceIDs []string
	Filter    *RetrieveFilter
}

// DetectContentFormat classifies a source from the explicit format, filename,
// and MIME content type.
func DetectContentFormat(format ContentFormat, filename, contentType string) ContentFormat {
	switch format {
	case ContentFormatMarkdown, ContentFormatPlain:
		return format
	}

	ext := strings.ToLower(filepath.Ext(filename))
	if ext == ".md" || ext == ".markdown" || ext == ".mdown" {
		return ContentFormatMarkdown
	}

	mime := strings.ToLower(strings.TrimSpace(strings.Split(contentType, ";")[0]))
	if mime == "text/markdown" || mime == "text/x-markdown" {
		return ContentFormatMarkdown
	}

	return ContentFormatPlain
}

// DocumentsFromSource converts a raw source into LangChain-style documents:
// page content plus metadata. Markdown is split into heading-aware section
// documents before chunking; plain text remains a single document.
func DocumentsFromSource(source Source) ([]Document, ContentFormat) {
	format := DetectContentFormat(source.Format, source.Filename, source.ContentType)
	text := strings.TrimSpace(source.Text)
	if text == "" {
		return nil, format
	}

	metadata := cloneMetadata(source.Metadata)
	if filename := strings.TrimSpace(source.Filename); filename != "" && metadata[qdrantclient.FieldSourceName] == "" {
		metadata[qdrantclient.FieldSourceName] = filename
	}

	if format == ContentFormatMarkdown {
		return ParseMarkdownDocuments(text, metadata), format
	}

	metadata[qdrantclient.FieldContentType] = PlainContentType
	return []Document{{
		PageContent: text,
		Metadata:    metadata,
	}}, format
}
