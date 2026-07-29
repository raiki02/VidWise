package tools

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/raiki02/vidwise/internal/search"
)

func TestWebSearchToolExposesOnlySimpleArguments(t *testing.T) {
	tool, err := NewWebSearchTool(&fakeSearchService{})
	if err != nil {
		t.Fatalf("NewWebSearchTool() error = %v", err)
	}
	info, err := tool.Info(context.Background())
	if err != nil {
		t.Fatalf("Info() error = %v", err)
	}
	body, err := json.Marshal(info)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	schemaText := string(body)
	for _, want := range []string{"query", "freshness"} {
		if !strings.Contains(schemaText, want) {
			t.Fatalf("schema missing %q: %s", want, schemaText)
		}
	}
	for _, forbidden := range []string{"provider", "crawler_depth", "rerank_model", "timeout", "max_chunk"} {
		if strings.Contains(schemaText, forbidden) {
			t.Fatalf("schema exposed forbidden argument %q: %s", forbidden, schemaText)
		}
	}
}

func TestWebSearchToolCallsSearchService(t *testing.T) {
	service := &fakeSearchService{
		result: &search.SearchResult{
			Summary: "ok",
			Sources: []search.Source{{
				Title:   "source",
				URL:     "https://example.com",
				Content: "content",
			}},
		},
	}
	tool, err := NewWebSearchTool(service)
	if err != nil {
		t.Fatalf("NewWebSearchTool() error = %v", err)
	}

	out, err := tool.InvokableRun(context.Background(), `{"query":"  OpenAI latest  ","freshness":"latest"}`)
	if err != nil {
		t.Fatalf("InvokableRun() error = %v", err)
	}
	if service.query != "OpenAI latest 时间要求：latest" {
		t.Fatalf("service query = %q", service.query)
	}
	var decoded search.SearchResult
	if err := json.Unmarshal([]byte(out), &decoded); err != nil {
		t.Fatalf("output is not SearchResult JSON: %v\n%s", err, out)
	}
	if decoded.Summary != "ok" || len(decoded.Sources) != 1 {
		t.Fatalf("unexpected output: %#v", decoded)
	}
}

func TestWebSearchToolRejectsInvalidArgumentsBeforeCallingService(t *testing.T) {
	service := &fakeSearchService{}
	tool, err := NewWebSearchTool(service)
	if err != nil {
		t.Fatalf("NewWebSearchTool() error = %v", err)
	}

	if _, err := tool.InvokableRun(context.Background(), `{"query":"  "}`); err == nil {
		t.Fatal("expected empty query error")
	}
	if service.query != "" {
		t.Fatalf("service should not be called for empty query, got %q", service.query)
	}

	tooLongFreshness := strings.Repeat("x", 65)
	payload, err := json.Marshal(WebSearchInput{Query: "OpenAI", Freshness: tooLongFreshness})
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	if _, err := tool.InvokableRun(context.Background(), string(payload)); err == nil {
		t.Fatal("expected overlong freshness error")
	}
}

func TestWebSearchToolPropagatesServiceErrors(t *testing.T) {
	tool, err := NewWebSearchTool(&fakeSearchService{err: errors.New("search unavailable")})
	if err != nil {
		t.Fatalf("NewWebSearchTool() error = %v", err)
	}

	if _, err := tool.InvokableRun(context.Background(), `{"query":"OpenAI"}`); err == nil || !strings.Contains(err.Error(), "search unavailable") {
		t.Fatalf("InvokableRun() error = %v, want service error", err)
	}
}

type fakeSearchService struct {
	query  string
	result *search.SearchResult
	err    error
}

func (s *fakeSearchService) Search(_ context.Context, query string) (*search.SearchResult, error) {
	s.query = query
	if s.err != nil {
		return nil, s.err
	}
	if s.result != nil {
		return s.result, nil
	}
	return &search.SearchResult{Summary: "empty", Sources: nil}, nil
}
