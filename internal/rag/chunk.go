package rag

import "strings"

// Split chunks text into overlapping rune windows of size runes with
// overlapRunes of shared context between adjacent chunks. Runs of whitespace
// are collapsed. Empty or whitespace-only text returns nil.
func Split(text string, size, overlap int) []string {
	text = strings.TrimSpace(normalizeWhiteSpace(text))
	if text == "" || size <= 0 {
		return nil
	}
	if overlap < 0 || overlap >= size {
		overlap = size / 5
	}
	if overlap > size {
		overlap = size
	}

	runes := []rune(text)
	if len(runes) <= size {
		return []string{string(runes)}
	}

	step := size - overlap
	if step <= 0 {
		step = size
	}

	chunks := make([]string, 0, (len(runes)+step-1)/step)
	for start := 0; start < len(runes); {
		end := start + size
		if end > len(runes) {
			end = len(runes)
		}
		chunk := strings.TrimSpace(string(runes[start:end]))
		if chunk != "" {
			chunks = append(chunks, chunk)
		}
		if end == len(runes) {
			break
		}
		start += step
	}
	return chunks
}

func normalizeWhiteSpace(text string) string {
	return strings.Join(strings.Fields(text), " ")
}
