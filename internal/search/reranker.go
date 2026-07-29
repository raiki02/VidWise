package search

import (
	"math"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

type KeywordReranker struct{}

func NewKeywordReranker() *KeywordReranker {
	return &KeywordReranker{}
}

func (r *KeywordReranker) Rank(query string, docs []Document) []Document {
	out := append([]Document(nil), docs...)
	terms := queryTerms(query)
	for i := range out {
		out[i].Score = scoreDocument(terms, out[i])
	}
	sort.SliceStable(out, func(i, j int) bool {
		return out[i].Score > out[j].Score
	})
	return out
}

func scoreDocument(terms []string, doc Document) float64 {
	haystack := strings.ToLower(doc.Title + " " + doc.Content)
	score := math.Log(float64(utf8.RuneCountInString(doc.Content)+1)) / 10
	for _, term := range terms {
		if term == "" {
			continue
		}
		count := strings.Count(haystack, term)
		if count > 0 {
			score += float64(count) * 5
		}
		if strings.Contains(strings.ToLower(doc.Title), term) {
			score += 3
		}
	}
	return score
}

func queryTerms(query string) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0)
	for _, part := range strings.FieldsFunc(strings.ToLower(query), func(r rune) bool {
		return unicode.IsSpace(r) || unicode.IsPunct(r) || unicode.IsSymbol(r)
	}) {
		part = strings.TrimSpace(part)
		if len([]rune(part)) < 2 {
			continue
		}
		if _, ok := seen[part]; ok {
			continue
		}
		seen[part] = struct{}{}
		out = append(out, part)
	}
	return out
}
