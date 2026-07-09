package chunk

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// Format describes the input text structure the splitter should preserve.
type Format string

const (
	FormatPlain    Format = "plain"
	FormatMarkdown Format = "markdown"
)

// Source describes the boundary that produced a chunk.
type Source string

const (
	SourceSection   Source = "section"
	SourceParagraph Source = "paragraph"
	SourceSentence  Source = "sentence"
	SourceClause    Source = "clause"
	SourceLine      Source = "line"
	SourceWord      Source = "word"
	SourceSliding   Source = "sliding"
)

// Chunk is a text segment with split metadata.
type Chunk struct {
	Index  int
	Text   string
	Source Source
}

// Config controls recursive text chunking.
type Config struct {
	MaxRunes     int
	MinRunes     int
	OverlapRunes int
	Format       Format
}

// DefaultConfig returns conservative defaults for embedding-oriented chunking.
func DefaultConfig() Config {
	return Config{
		MaxRunes:     1024,
		MinRunes:     20,
		OverlapRunes: 128,
		Format:       FormatPlain,
	}
}

type unit struct {
	text   string
	source Source
}

type splitLevel struct {
	source Source
	split  func(string) []string
}

// SplitText chunks text by preserving high-level structure first, then
// recursively splitting only the oversized units on progressively weaker
// natural boundaries.
func SplitText(text string, cfg Config) []Chunk {
	cfg = normalizeConfig(cfg)
	text = strings.TrimSpace(normalizeNewlines(text))
	if text == "" {
		return nil
	}

	units := splitTopLevelUnits(text, cfg.Format)
	if len(units) == 0 {
		units = []unit{{text: text, source: SourceParagraph}}
	}

	chunks := packUnits(units, cfg)
	if cfg.MinRunes > 0 {
		chunks = filterChunks(chunks, cfg.MinRunes)
	}
	for i := range chunks {
		chunks[i].Index = i
	}
	return chunks
}

func normalizeConfig(cfg Config) Config {
	if cfg.MaxRunes <= 0 {
		cfg.MaxRunes = DefaultConfig().MaxRunes
	}
	if cfg.OverlapRunes < 0 {
		cfg.OverlapRunes = 0
	}
	if cfg.OverlapRunes >= cfg.MaxRunes {
		cfg.OverlapRunes = cfg.MaxRunes / 8
	}
	if cfg.Format == "" {
		cfg.Format = FormatPlain
	}
	return cfg
}

func splitTopLevelUnits(text string, format Format) []unit {
	if format == FormatMarkdown {
		sections := splitMarkdownSections(text)
		out := make([]unit, 0, len(sections))
		for _, section := range sections {
			out = append(out, unit{text: section, source: SourceSection})
		}
		return out
	}

	paragraphs := splitParagraphs(text)
	out := make([]unit, 0, len(paragraphs))
	for _, paragraph := range paragraphs {
		out = append(out, unit{text: paragraph, source: SourceParagraph})
	}
	return out
}

func packUnits(units []unit, cfg Config) []Chunk {
	var chunks []Chunk
	var current []unit
	currentRunes := 0

	flush := func() {
		text, source := joinUnits(current)
		if text != "" {
			chunks = append(chunks, makeChunk(text, source))
		}
		current = nil
		currentRunes = 0
	}

	for _, u := range units {
		u.text = strings.TrimSpace(u.text)
		if u.text == "" {
			continue
		}

		uRunes := utf8.RuneCountInString(u.text)
		if uRunes > cfg.MaxRunes {
			flush()
			chunks = append(chunks, splitOversizedUnit(u, cfg)...)
			continue
		}

		sep := 0
		if currentRunes > 0 {
			sep = 2
		}
		if currentRunes+sep+uRunes > cfg.MaxRunes {
			flush()
		}
		current = append(current, u)
		currentRunes += sep + uRunes
	}
	flush()

	return chunks
}

func joinUnits(units []unit) (string, Source) {
	if len(units) == 0 {
		return "", ""
	}
	parts := make([]string, 0, len(units))
	source := units[0].source
	for _, u := range units {
		if text := strings.TrimSpace(u.text); text != "" {
			parts = append(parts, text)
		}
		if source != u.source {
			source = SourceParagraph
		}
	}
	return strings.TrimSpace(strings.Join(parts, "\n\n")), source
}

func splitOversizedUnit(u unit, cfg Config) []Chunk {
	if cfg.Format == FormatMarkdown {
		if chunks := splitOversizedMarkdownUnit(u.text, cfg); len(chunks) > 0 {
			return chunks
		}
	}
	return splitRecursive(u.text, cfg, []splitLevel{
		{source: SourceSentence, split: splitSentences},
		{source: SourceClause, split: splitClauses},
		{source: SourceLine, split: splitLines},
		{source: SourceWord, split: splitWords},
	})
}

func splitRecursive(text string, cfg Config, levels []splitLevel) []Chunk {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	if utf8.RuneCountInString(text) <= cfg.MaxRunes {
		source := SourceParagraph
		if len(levels) > 0 {
			source = levels[0].source
		}
		return []Chunk{makeChunk(text, source)}
	}
	if len(levels) == 0 {
		return slidingChunks(text, cfg.MaxRunes, cfg.OverlapRunes)
	}

	level := levels[0]
	segments := level.split(text)
	if len(segments) <= 1 {
		return splitRecursive(text, cfg, levels[1:])
	}

	var chunks []Chunk
	for _, segment := range packSegments(segments, cfg.MaxRunes) {
		if utf8.RuneCountInString(segment) <= cfg.MaxRunes {
			chunks = append(chunks, makeChunk(segment, level.source))
			continue
		}
		chunks = append(chunks, splitRecursive(segment, cfg, levels[1:])...)
	}
	if len(chunks) > 0 {
		return chunks
	}
	return splitRecursive(text, cfg, levels[1:])
}

func packSegments(segments []string, maxRunes int) []string {
	var packed []string
	current := strings.Builder{}
	currentRunes := 0

	flush := func() {
		text := strings.TrimSpace(current.String())
		if text != "" {
			packed = append(packed, text)
		}
		current.Reset()
		currentRunes = 0
	}

	for _, segment := range segments {
		segment = strings.TrimSpace(segment)
		if segment == "" {
			continue
		}
		segmentRunes := utf8.RuneCountInString(segment)
		if segmentRunes > maxRunes {
			flush()
			packed = append(packed, segment)
			continue
		}

		sep := 0
		if currentRunes > 0 {
			sep = 1
		}
		if currentRunes+sep+segmentRunes > maxRunes {
			flush()
		}
		if current.Len() > 0 {
			current.WriteByte(' ')
			currentRunes++
		}
		current.WriteString(segment)
		currentRunes += segmentRunes
	}
	flush()
	return packed
}

func splitOversizedMarkdownUnit(text string, cfg Config) []Chunk {
	blocks := splitMarkdownBlocks(text)
	if len(blocks) <= 1 {
		return splitRecursive(text, cfg, []splitLevel{
			{source: SourceSentence, split: splitSentences},
			{source: SourceClause, split: splitClauses},
			{source: SourceLine, split: splitLines},
			{source: SourceWord, split: splitWords},
		})
	}

	prefix := ""
	if isMarkdownHeading(blocks[0]) {
		prefix = blocks[0]
		blocks = blocks[1:]
	}

	blockUnits := make([]unit, 0, len(blocks))
	for _, block := range blocks {
		source := SourceParagraph
		if isMarkdownHeading(block) {
			source = SourceSection
		}
		blockUnits = append(blockUnits, unit{text: block, source: source})
	}

	bodyCfg := cfg
	if prefix != "" {
		available := cfg.MaxRunes - utf8.RuneCountInString(prefix) - 2
		if available > cfg.MaxRunes/3 {
			bodyCfg.MaxRunes = available
		} else {
			prefix = ""
		}
	}

	bodyChunks := packUnits(blockUnits, bodyCfg)
	if prefix == "" {
		return bodyChunks
	}

	chunks := make([]Chunk, 0, len(bodyChunks))
	for _, body := range bodyChunks {
		if strings.TrimSpace(body.Text) == "" {
			continue
		}
		chunks = append(chunks, makeChunk(prefix+"\n\n"+body.Text, SourceSection))
	}
	if len(chunks) == 0 {
		return []Chunk{makeChunk(prefix, SourceSection)}
	}
	return chunks
}

func splitMarkdownSections(text string) []string {
	blocks := splitMarkdownBlocks(text)
	if len(blocks) == 0 {
		return nil
	}

	var sections []string
	current := strings.Builder{}
	flush := func() {
		section := strings.TrimSpace(current.String())
		if section != "" {
			sections = append(sections, section)
		}
		current.Reset()
	}

	for _, block := range blocks {
		if isMarkdownHeading(block) {
			flush()
		}
		if current.Len() > 0 {
			current.WriteString("\n\n")
		}
		current.WriteString(block)
	}
	flush()
	return sections
}

func splitMarkdownBlocks(text string) []string {
	lines := strings.Split(normalizeNewlines(text), "\n")
	var blocks []string
	var current []string
	var inFence bool
	var fenceMarker string

	flush := func() {
		block := strings.TrimSpace(strings.Join(current, "\n"))
		if block != "" {
			blocks = append(blocks, block)
		}
		current = nil
	}

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if marker, ok := markdownFenceMarker(trimmed); ok {
			current = append(current, line)
			if !inFence {
				inFence = true
				fenceMarker = marker
			} else if marker == fenceMarker {
				inFence = false
				fenceMarker = ""
				flush()
			}
			continue
		}

		if inFence {
			current = append(current, line)
			continue
		}
		if trimmed == "" {
			flush()
			continue
		}
		if isMarkdownHeading(trimmed) || isMarkdownThematicBreak(trimmed) {
			flush()
			current = append(current, line)
			flush()
			continue
		}
		current = append(current, line)
	}
	flush()
	return blocks
}

func splitParagraphs(text string) []string {
	var paragraphs []string
	lines := strings.Split(normalizeNewlines(text), "\n")
	current := strings.Builder{}

	flush := func() {
		paragraph := strings.TrimSpace(current.String())
		if paragraph != "" {
			paragraphs = append(paragraphs, paragraph)
		}
		current.Reset()
	}

	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			flush()
			continue
		}
		if current.Len() > 0 {
			current.WriteByte('\n')
		}
		current.WriteString(line)
	}
	flush()
	return paragraphs
}

func splitSentences(text string) []string {
	return splitAfterRunes(strings.ReplaceAll(text, "\n", " "), isSentenceTerminator, true)
}

func splitClauses(text string) []string {
	return splitAfterRunes(text, isClauseBoundary, false)
}

func splitAfterRunes(text string, boundary func(rune) bool, includeTrailing bool) []string {
	runes := []rune(text)
	var parts []string
	start := 0

	for i, r := range runes {
		if !boundary(r) {
			continue
		}
		end := i + 1
		if includeTrailing {
			for end < len(runes) && isTrailingPunct(runes[end]) {
				end++
			}
		}
		part := strings.TrimSpace(string(runes[start:end]))
		if part != "" {
			parts = append(parts, part)
		}
		start = end
	}
	if start < len(runes) {
		remainder := strings.TrimSpace(string(runes[start:]))
		if remainder != "" {
			parts = append(parts, remainder)
		}
	}
	return parts
}

func splitLines(text string) []string {
	lines := strings.Split(normalizeNewlines(text), "\n")
	parts := make([]string, 0, len(lines))
	for _, line := range lines {
		if line = strings.TrimSpace(line); line != "" {
			parts = append(parts, line)
		}
	}
	return parts
}

func splitWords(text string) []string {
	return strings.Fields(text)
}

func slidingChunks(text string, maxRunes, overlapRunes int) []Chunk {
	runes := []rune(text)
	if len(runes) <= maxRunes {
		return []Chunk{makeChunk(text, SourceSliding)}
	}

	var chunks []Chunk
	for start := 0; start < len(runes); {
		end := start + maxRunes
		if end > len(runes) {
			end = len(runes)
		}
		chunkText := strings.TrimSpace(string(runes[start:end]))
		if chunkText != "" {
			chunks = append(chunks, makeChunk(chunkText, SourceSliding))
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

func filterChunks(chunks []Chunk, minRunes int) []Chunk {
	var kept []Chunk
	for _, c := range chunks {
		if !isLowQuality(c.Text, minRunes) {
			kept = append(kept, c)
		}
	}
	if len(kept) > 0 {
		return kept
	}

	nonEmpty := chunks[:0]
	for _, c := range chunks {
		if strings.TrimSpace(c.Text) != "" {
			nonEmpty = append(nonEmpty, c)
		}
	}
	return nonEmpty
}

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
		if unicode.IsLetter(r) || unicode.IsDigit(r) || unicode.Is(unicode.Han, r) {
			meaningful++
		}
	}
	return float64(meaningful)/float64(len(runes)) < 0.30
}

func makeChunk(text string, source Source) Chunk {
	return Chunk{Text: strings.TrimSpace(text), Source: source}
}

func normalizeNewlines(text string) string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	return strings.ReplaceAll(text, "\r", "\n")
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

func isClauseBoundary(r rune) bool {
	switch r {
	case '，', ',', '；', ';', '：', ':', '、':
		return true
	}
	return false
}

func isMarkdownHeading(block string) bool {
	line := strings.TrimSpace(firstLine(block))
	if line == "" || line[0] != '#' {
		return false
	}
	count := 0
	for _, r := range line {
		if r != '#' {
			break
		}
		count++
	}
	return count >= 1 && count <= 6 && len(line) > count && line[count] == ' '
}

func firstLine(text string) string {
	if idx := strings.IndexByte(text, '\n'); idx >= 0 {
		return text[:idx]
	}
	return text
}

func markdownFenceMarker(line string) (string, bool) {
	if strings.HasPrefix(line, "```") {
		return "```", true
	}
	if strings.HasPrefix(line, "~~~") {
		return "~~~", true
	}
	return "", false
}

func isMarkdownThematicBreak(line string) bool {
	line = strings.ReplaceAll(strings.TrimSpace(line), " ", "")
	if len(line) < 3 {
		return false
	}
	for _, marker := range []rune{'-', '*', '_'} {
		matched := true
		for _, r := range line {
			if r != marker {
				matched = false
				break
			}
		}
		if matched {
			return true
		}
	}
	return false
}
