package rag

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// ContextConfig controls how retrieved chunks are packed into an LLM prompt.
type ContextConfig struct {
	MaxRunes int
}

// DefaultContextConfig returns a conservative prompt-context budget for RAG
// snippets. The final prompt also contains system text, chat history, and user
// memory, so this should stay below the model's full context window.
func DefaultContextConfig() ContextConfig {
	return ContextConfig{MaxRunes: 12000}
}

// PackedContext is the prompt-ready representation of retrieved chunks.
type PackedContext struct {
	Text       string
	UsedChunks int
	Truncated  bool
}

func (c PackedContext) HasContext() bool {
	return strings.TrimSpace(c.Text) != ""
}

// PackContext formats retrieved chunks as numbered, source-labelled snippets
// and enforces a rune budget before the prompt is sent to the LLM.
func PackContext(chunks []RelevantChunk, cfg ContextConfig) PackedContext {
	cfg = normalizeContextConfig(cfg)
	if len(chunks) == 0 {
		return PackedContext{}
	}

	var out PackedContext
	var b strings.Builder
	usedRunes := 0
	snippetNumber := 0
	separator := "\n---\n"

	for _, chunk := range chunks {
		body := strings.TrimSpace(chunk.Text)
		if body == "" {
			continue
		}

		snippetNumber++
		prefix := ""
		if b.Len() > 0 {
			prefix = separator
		}

		remaining := cfg.MaxRunes - usedRunes - runeLen(prefix)
		if remaining <= 0 {
			out.Truncated = true
			break
		}

		entry := formatContextEntry(snippetNumber, chunk, body)
		entryRunes := runeLen(entry)
		if entryRunes > remaining {
			entry = truncateContextEntry(snippetNumber, chunk, body, remaining)
			if entry == "" {
				out.Truncated = true
				break
			}
			entryRunes = runeLen(entry)
			out.Truncated = true
		}

		b.WriteString(prefix)
		b.WriteString(entry)
		usedRunes += runeLen(prefix) + entryRunes
		out.UsedChunks++

		if out.Truncated {
			break
		}
	}

	out.Text = b.String()
	return out
}

// FormatChunkSource returns a stable human-readable citation label.
func FormatChunkSource(chunk RelevantChunk) string {
	parts := make([]string, 0, 4)
	if chunk.SourceName != "" {
		parts = append(parts, chunk.SourceName)
	}
	if chunk.HeadingPath != "" {
		parts = append(parts, chunk.HeadingPath)
	} else if chunk.DocumentTitle != "" {
		parts = append(parts, chunk.DocumentTitle)
	}
	if chunk.SourceURL != "" {
		parts = append(parts, chunk.SourceURL)
	}
	if chunk.TaskID != "" {
		parts = append(parts, "task:"+chunk.TaskID)
	}
	if chunk.ChunkID != "" {
		parts = append(parts, "chunk:"+chunk.ChunkID)
	}
	return strings.Join(parts, " / ")
}

func normalizeContextConfig(cfg ContextConfig) ContextConfig {
	defaults := DefaultContextConfig()
	if cfg.MaxRunes <= 0 {
		cfg.MaxRunes = defaults.MaxRunes
	}
	return cfg
}

func formatContextEntry(index int, chunk RelevantChunk, body string) string {
	header := formatContextHeader(index, chunk)
	if strings.TrimSpace(body) == "" {
		return header
	}
	return strings.TrimSpace(header + "\n" + strings.TrimSpace(body))
}

func formatContextHeader(index int, chunk RelevantChunk) string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("[片段 %d]\n", index))
	if source := FormatChunkSource(chunk); source != "" {
		b.WriteString("来源: ")
		b.WriteString(source)
		b.WriteString("\n")
	}
	b.WriteString(fmt.Sprintf("相关度: %.3f\n", chunk.Score))
	return strings.TrimSpace(b.String())
}

func truncateContextEntry(index int, chunk RelevantChunk, body string, maxRunes int) string {
	suffix := "\n[片段已截断]"
	header := formatContextHeader(index, chunk)

	available := maxRunes - runeLen(header) - runeLen("\n") - runeLen(suffix)
	if available <= 0 {
		return ""
	}
	truncatedBody := truncateRunes(strings.TrimSpace(body), available)
	if truncatedBody == "" {
		return ""
	}
	return strings.TrimSpace(header + "\n" + truncatedBody + suffix)
}

func truncateRunes(text string, maxRunes int) string {
	if maxRunes <= 0 {
		return ""
	}
	if runeLen(text) <= maxRunes {
		return text
	}
	runes := []rune(text)
	return strings.TrimSpace(string(runes[:maxRunes]))
}

func runeLen(text string) int {
	return utf8.RuneCountInString(text)
}
