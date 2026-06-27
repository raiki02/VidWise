package rag

import (
	"strings"
	"unicode"
	"unicode/utf8"
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
	ChunkSourceParagraph ChunkSource = "paragraph" // complete, naturally-bounded paragraph
	ChunkSourceSentence  ChunkSource = "sentence"  // split at sentence terminators
	ChunkSourceClause    ChunkSource = "clause"     // split at clause boundaries (comma etc.)
	ChunkSourceSliding   ChunkSource = "sliding"    // last-resort sliding window
)

// ChunkConfig controls text chunking behaviour.
type ChunkConfig struct {
	MaxRunes     int // preferred maximum runes per chunk (default 1024)
	MinRunes     int // drop chunks shorter than this unless they're the only one (default 64)
	OverlapRunes int // overlap between sliding-window chunks (default 128)
}

// DefaultChunkConfig returns sensible defaults.
func DefaultChunkConfig() ChunkConfig {
	return ChunkConfig{
		MaxRunes:     1024,
		MinRunes:     20,  // low enough to keep a single CJK sentence (~10-30 runes)
		OverlapRunes: 128,
	}
}

// ChunkText splits text into semantic chunks for embedding.
//
// Strategy (production-grade, in priority order):
//  1. Split by blank lines (natural paragraphs) — best semantic unit.
//  2. Over-long paragraphs: split at sentence boundaries (。！？!?.…).
//  3. Still over-long: split at clause boundaries (，,；;：:、).
//  4. Still over-long: sliding window as last resort.
//  5. Quality filter: drop chunks that are too short or have <30% meaningful runes.
//     NEVER drops all chunks — at minimum, every non-empty chunk is kept.
//  6. Every chunk records its source for observability.
func ChunkText(text string, maxRunes, overlapRunes int) []TextChunk {
	cfg := DefaultChunkConfig()
	if maxRunes > 0 {
		cfg.MaxRunes = maxRunes
	}
	if overlapRunes > 0 {
		cfg.OverlapRunes = overlapRunes
	}
	if cfg.OverlapRunes >= cfg.MaxRunes {
		cfg.OverlapRunes = cfg.MaxRunes / 8
	}

	return chunkTextWithConfig(text, cfg)
}

func chunkTextWithConfig(text string, cfg ChunkConfig) []TextChunk {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	totalRunes := utf8.RuneCountInString(text)

	// Short text: return as-is if meaningful.
	if totalRunes <= cfg.MaxRunes {
		if !isLowQuality(text, cfg.MinRunes) {
			return []TextChunk{{Index: 0, Text: text, Source: ChunkSourceParagraph}}
		}
		return nil
	}

	// Step 1 — split by blank-line paragraphs.
	paragraphs := splitParagraphs(text)

	// Step 2 — split over-long paragraphs at semantic boundaries.
	var chunks []TextChunk
	for _, para := range paragraphs {
		chunks = append(chunks, splitParagraph(para, cfg)...)
	}

	// Step 3 — quality filter.
	chunks = filterChunks(chunks, cfg.MinRunes)

	// Step 4 — re-index after filtering.
	for i := range chunks {
		chunks[i].Index = i
	}

	return chunks
}

// splitParagraph splits a single paragraph into one or more chunks.
func splitParagraph(text string, cfg ChunkConfig) []TextChunk {
	paraRunes := utf8.RuneCountInString(strings.TrimSpace(text))
	if paraRunes <= cfg.MaxRunes {
		if isLowQuality(text, cfg.MinRunes) {
			return nil
		}
		return []TextChunk{{Index: 0, Text: strings.TrimSpace(text), Source: ChunkSourceParagraph}}
	}

	// Over-long paragraph — try sentence splitting first.
	sentences := splitSentences(text)
	if len(sentences) > 1 {
		merged := mergeSegments(sentences, cfg, "sentence")
		var chunks []TextChunk
		for _, s := range merged {
			if isLowQuality(s, cfg.MinRunes) {
				continue
			}
			chunks = append(chunks, makeChunk(len(chunks), s, ChunkSourceSentence))
		}
		if len(chunks) > 0 {
			return chunks
		}
	}

	// Try clause splitting.
	clauses := splitClauses(text)
	if len(clauses) > 1 {
		merged := mergeSegments(clauses, cfg, "clause")
		var chunks []TextChunk
		for _, s := range merged {
			if isLowQuality(s, cfg.MinRunes) {
				continue
			}
			chunks = append(chunks, makeChunk(len(chunks), s, ChunkSourceClause))
		}
		if len(chunks) > 0 {
			return chunks
		}
	}

	// Last resort — sliding window. Clauses came back as one piece (no
	// clause boundaries found), so we fall through here.
	var chunks []TextChunk
	for _, sub := range slidingChunks(text, cfg.MaxRunes, cfg.OverlapRunes) {
		if isLowQuality(sub, cfg.MinRunes) {
			continue
		}
		chunks = append(chunks, makeChunk(len(chunks), sub, ChunkSourceSliding))
	}
	return chunks
}

// ------ Sentence / clause splitting ------------------------------------------

// splitSentences splits text at Chinese and English sentence terminators.
func splitSentences(text string) []string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")
	text = strings.ReplaceAll(text, "\n", " ")

	var sentences []string
	runes := []rune(text)
	start := 0

	for i, r := range runes {
		if isSentenceTerminator(r) {
			end := i + 1
			for end < len(runes) && isTrailingPunct(runes[end]) {
				end++
			}
			s := strings.TrimSpace(string(runes[start:end]))
			if s != "" {
				sentences = append(sentences, s)
			}
			start = end
		}
	}

	if start < len(runes) {
		remainder := strings.TrimSpace(string(runes[start:]))
		if remainder != "" {
			sentences = append(sentences, remainder)
		}
	}

	return sentences
}

func isSentenceTerminator(r rune) bool {
	switch r {
	case '。', '！', '？', '!', '?', '.', '…':
		return true
	}
	return false
}

func isTrailingPunct(r rune) bool {
	switch r {
	case '"', '」', '』', '）', ')', '】', '》', '>', '\'', '’':
		return true
	}
	return false
}

// splitClauses splits text at clause boundaries WITHOUT calling back into
// mergeSegments. This prevents the infinite recursion that would occur when
// a text segment has no sentence terminators AND no clause boundaries.
func splitClauses(text string) []string {
	runes := []rune(text)
	var clauses []string
	start := 0

	for i, r := range runes {
		if isClauseBoundary(r) {
			end := i + 1
			cl := strings.TrimSpace(string(runes[start:end]))
			if cl != "" {
				clauses = append(clauses, cl)
			}
			start = end
		}
	}
	if start < len(runes) {
		remainder := strings.TrimSpace(string(runes[start:]))
		if remainder != "" {
			clauses = append(clauses, remainder)
		}
	}

	return clauses
}

func isClauseBoundary(r rune) bool {
	switch r {
	case '，', ',', '；', ';', '：', ':', '、':
		return true
	}
	return false
}

// mergeSegments merges consecutive text segments whose combined length stays
// within maxRunes, preventing tiny singleton chunks. Segments that are still
// too large after merging are kept as-is (they will be handled by a
// higher-level split strategy, e.g. clause split or sliding window).
func mergeSegments(segments []string, cfg ChunkConfig, _ string) []string {
	if len(segments) == 0 {
		return nil
	}

	var merged []string
	buf := strings.Builder{}
	bufRunes := 0

	flush := func() {
		s := strings.TrimSpace(buf.String())
		if s != "" {
			merged = append(merged, s)
		}
		buf.Reset()
		bufRunes = 0
	}

	for _, s := range segments {
		sRunes := utf8.RuneCountInString(s)

		// If this segment alone exceeds the limit, keep it — the caller
		// is responsible for trying a lower-level split strategy.
		flush()
		if sRunes > cfg.MaxRunes {
			merged = append(merged, s)
			continue
		}

		sep := 0
		if bufRunes > 0 {
			sep = 1 // space separator
		}

		if bufRunes+sep+sRunes > cfg.MaxRunes {
			flush()
		}
		if buf.Len() > 0 {
			buf.WriteByte(' ')
		}
		buf.WriteString(s)
		bufRunes += sep + sRunes
	}
	flush()

	if len(merged) == 0 {
		return segments
	}
	return merged
}

// ------ Quality filter -------------------------------------------------------

// isLowQuality returns true when a chunk is too degraded to index.
func isLowQuality(text string, minRunes int) bool {
	text = strings.TrimSpace(text)
	if text == "" {
		return true
	}

	runes := []rune(text)

	if len(runes) < minRunes {
		return true
	}

	meaningful := 0
	for _, r := range runes {
		if isMeaningfulRune(r) {
			meaningful++
		}
	}

	ratio := float64(meaningful) / float64(len(runes))
	if ratio < 0.30 {
		return true
	}

	return false
}

func isMeaningfulRune(r rune) bool {
	if unicode.IsLetter(r) || unicode.IsDigit(r) {
		return true
	}
	if unicode.Is(unicode.Han, r) {
		return true
	}
	return false
}

// filterChunks removes low-quality chunks. It never removes ALL chunks —
// if every chunk would be dropped it returns the originals (minus empty).
func filterChunks(chunks []TextChunk, minRunes int) []TextChunk {
	var kept []TextChunk
	for _, c := range chunks {
		if !isLowQuality(c.Text, minRunes) {
			kept = append(kept, c)
		}
	}

	if len(kept) == 0 {
		var nonEmpty []TextChunk
		for _, c := range chunks {
			if strings.TrimSpace(c.Text) != "" {
				nonEmpty = append(nonEmpty, c)
			}
		}
		return nonEmpty
	}
	return kept
}

// ------ Paragraph splitting --------------------------------------------------

func splitParagraphs(text string) []string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")

	var paragraphs []string
	lines := strings.Split(text, "\n")
	current := strings.Builder{}

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
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

// ------ Sliding window (last-resort fallback) -------------------------------

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
			next = start + maxRunes
		}
		if next >= len(runes) {
			break
		}
		start = next
	}
	return chunks
}

func makeChunk(idx int, text string, source ChunkSource) TextChunk {
	return TextChunk{
		Index:  idx,
		Text:   strings.TrimSpace(text),
		Source: source,
	}
}
