package rag

import (
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

var (
	pageMarkerPattern          = regexp.MustCompile(`(?i)^(?:page\s*)?\d+\s*(?:/|of)\s*\d+$`)
	pageMarkerWithWordPattern  = regexp.MustCompile(`(?i)^page\s+\d+$`)
	chinesePageMarkerPattern   = regexp.MustCompile(`^第\s*\d+\s*页(?:\s*/\s*共?\s*\d+\s*页)?$`)
	hyphenPageMarkerPattern    = regexp.MustCompile(`^[-–—]\s*\d+\s*[-–—]$`)
	markdownFrontMatterYAMLKey = regexp.MustCompile(`^\s*[A-Za-z0-9_-]+\s*:`)
	markdownFrontMatterTOMLKey = regexp.MustCompile(`^\s*[A-Za-z0-9_.-]+\s*=`)
)

// CleanSourceText normalizes and filters raw knowledge text before it is parsed,
// chunked, embedded, and stored. It is intentionally conservative: it removes
// common ingestion noise without rewriting meaningful document content.
func CleanSourceText(text string, format ContentFormat) string {
	switch format {
	case ContentFormatMarkdown:
		return cleanMarkdownText(text)
	default:
		return cleanPlainText(text)
	}
}

func cleanPlainText(text string) string {
	lines := strings.Split(normalizeTextForCleaning(text), "\n")
	cleaned := make([]string, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(collapseHorizontalWhitespace(line))
		if line == "" {
			cleaned = append(cleaned, "")
			continue
		}
		if isPlainNoiseLine(line) {
			continue
		}
		cleaned = append(cleaned, line)
	}

	return strings.TrimSpace(strings.Join(collapseBlankLines(unwrapPlainLines(cleaned)), "\n"))
}

func cleanMarkdownText(text string) string {
	text = stripMarkdownFrontMatter(normalizeTextForCleaning(text))
	lines := strings.Split(text, "\n")

	out := make([]string, 0, len(lines))
	inFence := false
	fenceMarker := ""
	inComment := false
	inDroppedHTMLBlock := ""
	blankPending := false

	flushBlank := func() {
		if blankPending && len(out) > 0 {
			out = append(out, "")
		}
		blankPending = false
	}

	for _, rawLine := range lines {
		line := strings.TrimRight(rawLine, " \t")
		trimmed := strings.TrimSpace(line)

		if marker, ok := markdownFenceMarkerForCleaning(trimmed); ok {
			flushBlank()
			out = append(out, line)
			if !inFence {
				inFence = true
				fenceMarker = marker
			} else if marker == fenceMarker {
				inFence = false
				fenceMarker = ""
			}
			continue
		}

		if inFence {
			out = append(out, rawLine)
			continue
		}

		if inDroppedHTMLBlock != "" {
			if strings.Contains(strings.ToLower(trimmed), "</"+inDroppedHTMLBlock+">") {
				inDroppedHTMLBlock = ""
			}
			continue
		}

		if tag, drop := markdownHTMLBlockToDrop(trimmed); drop {
			if !strings.Contains(strings.ToLower(trimmed), "</"+tag+">") {
				inDroppedHTMLBlock = tag
			}
			continue
		}

		line = stripHTMLComments(line, &inComment)
		trimmed = strings.TrimSpace(line)
		if trimmed == "" {
			blankPending = true
			continue
		}
		if isMarkdownReferenceDefinition(line) {
			continue
		}

		flushBlank()
		out = append(out, line)
	}

	return strings.TrimSpace(strings.Join(out, "\n"))
}

func normalizeTextForCleaning(text string) string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")
	text = strings.TrimPrefix(text, "\ufeff")

	return strings.Map(func(r rune) rune {
		switch r {
		case '\ufeff', '\u200b', '\u200c', '\u200d', '\u2060':
			return -1
		case '\u00a0', '\u2007', '\u202f':
			return ' '
		case '\n', '\t':
			return r
		case '\f', '\v':
			return '\n'
		}
		if unicode.IsControl(r) {
			return -1
		}
		return r
	}, text)
}

func collapseHorizontalWhitespace(text string) string {
	return strings.Join(strings.Fields(strings.ReplaceAll(text, "\t", " ")), " ")
}

func collapseBlankLines(lines []string) []string {
	out := make([]string, 0, len(lines))
	lastBlank := true
	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			if !lastBlank {
				out = append(out, "")
				lastBlank = true
			}
			continue
		}
		out = append(out, line)
		lastBlank = false
	}
	for len(out) > 0 && strings.TrimSpace(out[len(out)-1]) == "" {
		out = out[:len(out)-1]
	}
	return out
}

func unwrapPlainLines(lines []string) []string {
	out := make([]string, 0, len(lines))
	block := make([]string, 0, 4)

	flush := func() {
		out = append(out, block...)
		block = block[:0]
	}

	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			flush()
			out = append(out, "")
			continue
		}
		if len(block) > 0 && shouldJoinPlainLines(block[len(block)-1], line) {
			block[len(block)-1] = joinPlainLines(block[len(block)-1], line)
			continue
		}
		block = append(block, line)
	}
	flush()

	return out
}

func shouldJoinPlainLines(prev, next string) bool {
	prev = strings.TrimSpace(prev)
	next = strings.TrimSpace(next)
	if prev == "" || next == "" {
		return false
	}
	if isListLikeLine(prev) || isListLikeLine(next) || isTableLikeLine(prev) || isTableLikeLine(next) {
		return false
	}
	if strings.HasSuffix(prev, "-") && startsWithLowerLatin(next) {
		return true
	}
	if shouldJoinWithoutSpace(prev, next) && utf8.RuneCountInString(prev) >= 8 {
		return true
	}
	if endsWithTerminalPunctuation(prev) {
		return false
	}
	if utf8.RuneCountInString(prev) < 12 && !strings.ContainsAny(prev, ",，、") {
		return false
	}
	return true
}

func joinPlainLines(prev, next string) string {
	prev = strings.TrimSpace(prev)
	next = strings.TrimSpace(next)
	if strings.HasSuffix(prev, "-") && startsWithLowerLatin(next) {
		return strings.TrimSuffix(prev, "-") + next
	}
	if shouldJoinWithoutSpace(prev, next) {
		return prev + next
	}
	return prev + " " + next
}

func shouldJoinWithoutSpace(prev, next string) bool {
	last, _ := utf8.DecodeLastRuneInString(prev)
	first, _ := utf8.DecodeRuneInString(next)
	return isCJKRune(last) && isCJKRune(first)
}

func endsWithTerminalPunctuation(text string) bool {
	text = strings.TrimSpace(text)
	if text == "" {
		return false
	}
	r, _ := utf8.DecodeLastRuneInString(text)
	switch r {
	case '.', '!', '?', ';', ':', '。', '！', '？', '；', '：', '…':
		return true
	default:
		return false
	}
}

func startsWithLowerLatin(text string) bool {
	r, _ := utf8.DecodeRuneInString(strings.TrimSpace(text))
	return r >= 'a' && r <= 'z'
}

func isCJKRune(r rune) bool {
	return unicode.Is(unicode.Han, r) || unicode.Is(unicode.Hiragana, r) || unicode.Is(unicode.Katakana, r) || unicode.Is(unicode.Hangul, r)
}

func isListLikeLine(line string) bool {
	line = strings.TrimSpace(line)
	if line == "" {
		return false
	}
	if strings.HasPrefix(line, "- ") || strings.HasPrefix(line, "* ") || strings.HasPrefix(line, "+ ") || strings.HasPrefix(line, "• ") {
		return true
	}
	if strings.HasPrefix(line, "[") || strings.HasPrefix(line, "(") {
		return true
	}
	runes := []rune(line)
	digitEnd := 0
	for digitEnd < len(runes) && unicode.IsDigit(runes[digitEnd]) {
		digitEnd++
	}
	if digitEnd > 0 && digitEnd+1 < len(runes) {
		marker := runes[digitEnd]
		if (marker == '.' || marker == ')' || marker == '、') && unicode.IsSpace(runes[digitEnd+1]) {
			return true
		}
	}
	return false
}

func isTableLikeLine(line string) bool {
	return strings.Count(line, "|") >= 2
}

func isPlainNoiseLine(line string) bool {
	if pageMarkerPattern.MatchString(line) ||
		pageMarkerWithWordPattern.MatchString(line) ||
		chinesePageMarkerPattern.MatchString(line) ||
		hyphenPageMarkerPattern.MatchString(line) {
		return true
	}
	if isSeparatorNoiseLine(line) {
		return true
	}
	return isLowMeaningLine(line)
}

func isSeparatorNoiseLine(line string) bool {
	line = strings.TrimSpace(line)
	if utf8.RuneCountInString(line) < 3 {
		return false
	}
	meaningful := 0
	for _, r := range line {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || unicode.Is(unicode.Han, r) {
			meaningful++
		}
	}
	if meaningful > 0 {
		return false
	}

	withoutSpaces := strings.ReplaceAll(line, " ", "")
	withoutSpaces = strings.ReplaceAll(withoutSpaces, "\t", "")
	if withoutSpaces == "" {
		return true
	}
	first, _ := utf8.DecodeRuneInString(withoutSpaces)
	for _, r := range withoutSpaces {
		if r != first {
			return false
		}
	}
	return true
}

func isLowMeaningLine(line string) bool {
	runes := []rune(strings.TrimSpace(line))
	if len(runes) < 8 {
		return false
	}
	meaningful := 0
	for _, r := range runes {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || unicode.Is(unicode.Han, r) {
			meaningful++
		}
	}
	return float64(meaningful)/float64(len(runes)) < 0.20
}

func stripMarkdownFrontMatter(text string) string {
	lines := strings.Split(text, "\n")
	start := 0
	for start < len(lines) && strings.TrimSpace(lines[start]) == "" {
		start++
	}
	if start >= len(lines) || start > 2 {
		return text
	}

	marker := strings.TrimSpace(lines[start])
	if marker != "---" && marker != "+++" {
		return text
	}

	for end := start + 1; end < len(lines) && end-start <= 120; end++ {
		endMarker := strings.TrimSpace(lines[end])
		if (marker == "---" && (endMarker == "---" || endMarker == "...")) || (marker == "+++" && endMarker == "+++") {
			if !looksLikeFrontMatter(lines[start+1 : end]) {
				return text
			}
			return strings.Join(lines[end+1:], "\n")
		}
	}
	return text
}

func looksLikeFrontMatter(lines []string) bool {
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if markdownFrontMatterYAMLKey.MatchString(line) || markdownFrontMatterTOMLKey.MatchString(line) {
			return true
		}
	}
	return false
}

func stripHTMLComments(line string, inComment *bool) string {
	var b strings.Builder
	for {
		if *inComment {
			end := strings.Index(line, "-->")
			if end < 0 {
				return b.String()
			}
			line = line[end+3:]
			*inComment = false
			continue
		}
		start := strings.Index(line, "<!--")
		if start < 0 {
			b.WriteString(line)
			return strings.TrimRight(b.String(), " \t")
		}
		b.WriteString(line[:start])
		line = line[start+4:]
		*inComment = true
	}
}

func isMarkdownReferenceDefinition(line string) bool {
	line = strings.TrimRight(line, " \t")
	trimmedLeft := strings.TrimLeft(line, " \t")
	if len(line)-len(trimmedLeft) > 3 || !strings.HasPrefix(trimmedLeft, "[") {
		return false
	}
	if strings.HasPrefix(trimmedLeft, "[^") {
		return false
	}
	close := strings.Index(trimmedLeft, "]:")
	if close <= 1 {
		return false
	}
	return strings.TrimSpace(trimmedLeft[close+2:]) != ""
}

func markdownHTMLBlockToDrop(line string) (string, bool) {
	lower := strings.ToLower(strings.TrimSpace(line))
	for _, tag := range []string{"script", "style", "noscript"} {
		if strings.HasPrefix(lower, "<"+tag) {
			return tag, true
		}
	}
	return "", false
}

func markdownFenceMarkerForCleaning(line string) (string, bool) {
	if strings.HasPrefix(line, "```") {
		return "```", true
	}
	if strings.HasPrefix(line, "~~~") {
		return "~~~", true
	}
	return "", false
}
