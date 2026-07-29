package search

import "unicode/utf8"

type BasicQualityEvaluator struct{}

func NewBasicQualityEvaluator() *BasicQualityEvaluator {
	return &BasicQualityEvaluator{}
}

func (e *BasicQualityEvaluator) Evaluate(query string, docs []Document) SearchQuality {
	if len(docs) == 0 {
		return SearchQuality{Score: 0, SourceCount: 0, Reason: "no_sources"}
	}
	totalRunes := 0
	for _, doc := range docs {
		totalRunes += utf8.RuneCountInString(doc.Content)
	}
	score := minFloat(1, float64(len(docs))/5)
	if totalRunes < 400 {
		score *= 0.6
	}
	reason := "sufficient_sources"
	if score < 0.5 {
		reason = "thin_sources"
	}
	return SearchQuality{
		Score:       score,
		SourceCount: len(docs),
		Reason:      reason,
	}
}

func minFloat(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}
