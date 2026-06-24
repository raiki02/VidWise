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

// ChunkText splits text into overlapping chunks for embedding.
func ChunkText(text string, chunkRunes, overlapRunes int) []TextChunk {
	if chunkRunes <= 0 {
		chunkRunes = 512
	}
	if overlapRunes < 0 {
		overlapRunes = 0
	}

	runes := []rune(text)
	total := len(runes)
	if total <= chunkRunes {
		return []TextChunk{{
			Index:     0,
			Text:      strings.TrimSpace(text),
			StartChar: 0,
			EndChar:   utf8.RuneCountInString(text),
		}}
	}

	var chunks []TextChunk
	for start := 0; start < total; {
		end := start + chunkRunes
		if end > total {
			end = total
		}
		chunkText := strings.TrimSpace(string(runes[start:end]))
		if chunkText != "" {
			chunks = append(chunks, TextChunk{
				Index:     len(chunks),
				Text:      chunkText,
				StartChar: start,
				EndChar:   end,
			})
		}
		start = end - overlapRunes
		if start >= total {
			break
		}
	}
	return chunks
}
