package search

import (
	"strings"
	"unicode/utf8"
)

type BasicCompressorConfig struct {
	MaxDocuments    int
	MaxContentRunes int
	MaxTotalRunes   int
}

type BasicCompressor struct {
	maxDocuments    int
	maxContentRunes int
	maxTotalRunes   int
}

func NewBasicCompressor(cfg BasicCompressorConfig) *BasicCompressor {
	if cfg.MaxDocuments <= 0 {
		cfg.MaxDocuments = 5
	}
	if cfg.MaxContentRunes <= 0 {
		cfg.MaxContentRunes = 1200
	}
	if cfg.MaxTotalRunes <= 0 {
		cfg.MaxTotalRunes = cfg.MaxDocuments * cfg.MaxContentRunes
	}
	return &BasicCompressor{
		maxDocuments:    cfg.MaxDocuments,
		maxContentRunes: cfg.MaxContentRunes,
		maxTotalRunes:   cfg.MaxTotalRunes,
	}
}

func (c *BasicCompressor) Compress(docs []Document) []Document {
	if c == nil {
		return docs
	}
	out := make([]Document, 0, min(len(docs), c.maxDocuments))
	total := 0
	for _, doc := range docs {
		if len(out) >= c.maxDocuments || total >= c.maxTotalRunes {
			break
		}
		doc.Content = strings.TrimSpace(doc.Content)
		if doc.Content == "" {
			continue
		}
		remaining := c.maxTotalRunes - total
		limit := min(c.maxContentRunes, remaining)
		if limit <= 0 {
			break
		}
		doc.Content = truncateRunes(doc.Content, limit)
		total += utf8.RuneCountInString(doc.Content)
		out = append(out, doc)
	}
	return out
}

func truncateRunes(text string, limit int) string {
	if limit <= 0 {
		return ""
	}
	if utf8.RuneCountInString(text) <= limit {
		return text
	}
	runes := []rune(text)
	if limit <= 3 {
		return string(runes[:limit])
	}
	return strings.TrimSpace(string(runes[:limit-3])) + "..."
}
