package rag

import (
	"strings"
	"unicode/utf8"
)

// TextChunk represents a segment of text with position metadata.
type TextChunk struct {
	Index     int    `json:"index"`
	Text      string `json:"text"`
	StartChar int    `json:"start_char"`
	EndChar   int    `json:"end_char"`
}

// ChunkText splits text into semantic chunks for embedding.
//
// Strategy (best-effort, in order):
//  1. Split by blank lines (natural paragraphs) — preserves semantic units.
//  2. If a paragraph exceeds maxRunes, use sliding window with overlap.
//     (No sentence-level splitting — it produces too many small fragments
//     that lack the surrounding context needed for good retrieval.)
func ChunkText(text string, maxRunes, overlapRunes int) []TextChunk {
	if maxRunes <= 0 {
		maxRunes = 512
	}
	if overlapRunes < 0 {
		overlapRunes = 0
	}
	if overlapRunes >= maxRunes {
		overlapRunes = maxRunes / 8
	}

	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	runes := []rune(text)
	if len(runes) <= maxRunes {
		return []TextChunk{{
			Index:     0,
			Text:      text,
			StartChar: 0,
			EndChar:   utf8.RuneCountInString(text),
		}}
	}

	// Step 1: split by blank-line paragraphs
	paragraphs := splitParagraphs(text)
	// Step 2: split over-long paragraphs with sliding window
	var chunks []TextChunk
	for _, para := range paragraphs {
		paraRunes := []rune(para)
		if len(paraRunes) <= maxRunes {
			chunks = append(chunks, makeChunk(len(chunks), para))
			continue
		}
		for _, sub := range slidingChunks(para, maxRunes, overlapRunes) {
			chunks = append(chunks, makeChunk(len(chunks), sub))
		}
	}

	// Compact empty chunks and re-index
	compact := make([]TextChunk, 0, len(chunks))
	for _, c := range chunks {
		if strings.TrimSpace(c.Text) != "" {
			c.Index = len(compact)
			compact = append(compact, c)
		}
	}
	return compact
}

// splitParagraphs splits text by blank lines (one or more consecutive newlines
// with optional whitespace).
func splitParagraphs(text string) []string {
	// Normalize: treat 2+ consecutive newlines (with optional spaces between) as paragraph break.
	// First normalize carriage returns.
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")

	var paragraphs []string
	lines := strings.Split(text, "\n")
	current := strings.Builder{}
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			// Blank line: flush current paragraph
			if current.Len() > 0 {
				paragraphs = append(paragraphs, strings.TrimSpace(current.String()))
				current.Reset()
			}
			continue
		}
		if current.Len() > 0 {
			current.WriteByte('\n')
		}
		current.WriteString(line)
	}
	if current.Len() > 0 {
		paragraphs = append(paragraphs, strings.TrimSpace(current.String()))
	}
	if len(paragraphs) == 0 {
		return []string{text}
	}
	return paragraphs
}

// slidingChunks splits text into fixed-rune chunks with overlap. Fallback
// when a paragraph exceeds maxRunes.
func slidingChunks(text string, maxRunes, overlapRunes int) []string {
	runes := []rune(text)
	if len(runes) <= maxRunes {
		return []string{text}
	}
	var chunks []string
	for start := 0; start < len(runes); {
		end := start + maxRunes
		if end > len(runes) {
			end = len(runes)
		}
		chunk := strings.TrimSpace(string(runes[start:end]))
		if chunk != "" {
			chunks = append(chunks, chunk)
		}
		next := end - overlapRunes
		if next <= start {
			next = start + maxRunes // ensure forward progress
		}
		if next >= len(runes) {
			break
		}
		start = next
	}
	return chunks
}

func makeChunk(idx int, text string) TextChunk {
	return TextChunk{
		Index:     idx,
		Text:      strings.TrimSpace(text),
		StartChar: 0,
		EndChar:   utf8.RuneCountInString(text),
	}
}
