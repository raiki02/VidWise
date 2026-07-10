package rag

import (
	"strings"

	"github.com/raiki02/vidwise/internal/chunk"
	qdrantclient "github.com/raiki02/vidwise/internal/storage/qdrant"
)

// TextChunk represents a segment of text with position metadata.
type TextChunk struct {
	Index int    `json:"index"`
	Text  string `json:"text"`
	// Source indicates how the chunk was produced.
	Source ChunkSource `json:"source"`
}

// ChunkSource describes the splitting strategy that produced this chunk.
type ChunkSource string

const (
	ChunkSourceSection   ChunkSource = "section"   // markdown/document section
	ChunkSourceParagraph ChunkSource = "paragraph" // complete, naturally-bounded paragraph
	ChunkSourceSentence  ChunkSource = "sentence"  // split at sentence terminators
	ChunkSourceClause    ChunkSource = "clause"    // split at clause boundaries
	ChunkSourceLine      ChunkSource = "line"      // split at line boundaries
	ChunkSourceWord      ChunkSource = "word"      // split at whitespace boundaries
	ChunkSourceSliding   ChunkSource = "sliding"   // last-resort sliding window
)

// ChunkConfig controls text chunking behaviour.
type ChunkConfig struct {
	MaxRunes     int // preferred maximum runes per chunk (default 1024)
	MinRunes     int // drop chunks shorter than this unless they're the only one (default 20)
	OverlapRunes int // overlap between sliding-window chunks (default 128)
}

// DefaultChunkConfig returns sensible defaults.
func DefaultChunkConfig() ChunkConfig {
	return ChunkConfig{
		MaxRunes:     1024,
		MinRunes:     20,
		OverlapRunes: 128,
	}
}

// ChunkText splits text into semantic chunks for embedding.
//
// Strategy:
//  1. Preserve document structure first: Markdown sections when present,
//     otherwise blank-line paragraphs.
//  2. Pack adjacent small units up to MaxRunes.
//  3. Recursively split only oversized units by sentence, clause, line, and
//     word boundaries.
//  4. Use sliding windows only as the final fallback.
//  5. Filter low-quality chunks without dropping all non-empty content.
func ChunkText(text string, maxRunes, overlapRunes int) []TextChunk {
	cfg := DefaultChunkConfig()
	if maxRunes > 0 {
		cfg.MaxRunes = maxRunes
	}
	if overlapRunes >= 0 {
		cfg.OverlapRunes = overlapRunes
	}
	return chunkTextWithConfig(text, cfg)
}

func chunkTextWithConfig(text string, cfg ChunkConfig) []TextChunk {
	return chunkTextWithFormat(text, cfg, chunk.FormatMarkdown)
}

func chunkDocumentTextWithConfig(text string, metadata map[string]string, cfg ChunkConfig) []TextChunk {
	format := chunk.FormatPlain
	if strings.EqualFold(strings.TrimSpace(metadata[qdrantclient.FieldContentType]), MarkdownContentType) {
		format = chunk.FormatMarkdown
	}
	return chunkTextWithFormat(text, cfg, format)
}

func chunkTextWithFormat(text string, cfg ChunkConfig, format chunk.Format) []TextChunk {
	chunks := chunk.SplitText(text, chunk.Config{
		MaxRunes:     cfg.MaxRunes,
		MinRunes:     cfg.MinRunes,
		OverlapRunes: cfg.OverlapRunes,
		Format:       format,
	})

	out := make([]TextChunk, 0, len(chunks))
	for _, c := range chunks {
		out = append(out, TextChunk{
			Index:  c.Index,
			Text:   c.Text,
			Source: mapChunkSource(c.Source),
		})
	}
	return out
}

func mapChunkSource(source chunk.Source) ChunkSource {
	switch source {
	case chunk.SourceSection:
		return ChunkSourceSection
	case chunk.SourceSentence:
		return ChunkSourceSentence
	case chunk.SourceClause:
		return ChunkSourceClause
	case chunk.SourceLine:
		return ChunkSourceLine
	case chunk.SourceWord:
		return ChunkSourceWord
	case chunk.SourceSliding:
		return ChunkSourceSliding
	default:
		return ChunkSourceParagraph
	}
}
