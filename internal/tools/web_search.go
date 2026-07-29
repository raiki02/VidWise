package tools

import (
	"context"
	"errors"
	"strings"

	einotool "github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
	"github.com/raiki02/vidwise/internal/search"
)

type WebSearchInput struct {
	Query     string `json:"query" jsonschema:"required" jsonschema_description:"Search keywords or a natural-language question."`
	Freshness string `json:"freshness,omitempty" jsonschema_description:"Optional time requirement such as today, latest, past week, or no preference."`
}

func NewWebSearchTool(service search.SearchService) (einotool.InvokableTool, error) {
	if service == nil {
		return nil, errors.New("search service is required")
	}
	return utils.InferTool(
		"web_search",
		"Search the web for current or external information. Input only query and optional freshness. Returns a concise summary and source snippets with URLs.",
		func(ctx context.Context, input WebSearchInput) (*search.SearchResult, error) {
			query := strings.TrimSpace(input.Query)
			if query == "" {
				return nil, errors.New("query is required")
			}
			freshness := strings.TrimSpace(input.Freshness)
			if err := validateFreshness(freshness); err != nil {
				return nil, err
			}
			return service.Search(ctx, renderQueryWithFreshness(query, freshness))
		},
	)
}

func validateFreshness(freshness string) error {
	if freshness == "" {
		return nil
	}
	if len([]rune(freshness)) > 64 {
		return errors.New("freshness is too long")
	}
	for _, r := range freshness {
		if r < 32 {
			return errors.New("freshness contains control characters")
		}
	}
	return nil
}

func renderQueryWithFreshness(query, freshness string) string {
	query = strings.TrimSpace(query)
	freshness = strings.TrimSpace(freshness)
	if freshness == "" {
		return query
	}
	return query + " 时间要求：" + freshness
}
