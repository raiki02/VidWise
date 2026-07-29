package search

import "testing"

func TestBasicQualityEvaluatorScoresEmptyAndThinSources(t *testing.T) {
	evaluator := NewBasicQualityEvaluator()
	empty := evaluator.Evaluate("query", nil)
	if empty.Score != 0 || empty.Reason != "no_sources" {
		t.Fatalf("empty quality = %#v", empty)
	}

	thin := evaluator.Evaluate("query", []Document{{Content: "short"}})
	if thin.Score <= 0 || thin.Reason != "thin_sources" {
		t.Fatalf("thin quality = %#v", thin)
	}
}
