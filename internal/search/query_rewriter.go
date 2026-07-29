package search

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	einomodel "github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
)

type ChatModelFactory func(context.Context) (QueryRewriteChatModel, error)

type QueryRewriteChatModel interface {
	Generate(ctx context.Context, input []*schema.Message, opts ...einomodel.Option) (*schema.Message, error)
}

type LLMQueryRewriter struct {
	newModel   ChatModelFactory
	maxQueries int
}

func NewLLMQueryRewriter(factory ChatModelFactory, maxQueries int) (*LLMQueryRewriter, error) {
	if factory == nil {
		return nil, errors.New("query rewrite model factory is required")
	}
	if maxQueries <= 0 {
		maxQueries = 3
	}
	return &LLMQueryRewriter{newModel: factory, maxQueries: maxQueries}, nil
}

func (r *LLMQueryRewriter) Rewrite(ctx context.Context, query string) ([]string, error) {
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, errors.New("query is required")
	}
	model, err := r.newModel(ctx)
	if err != nil {
		return nil, fmt.Errorf("create query rewrite model: %w", err)
	}
	resp, err := model.Generate(ctx, []*schema.Message{
		schema.SystemMessage(fmt.Sprintf(`你是搜索查询改写器。将用户问题改写为适合 Web Search 的查询。
要求：
1. 返回 JSON 字符串数组。
2. 最多返回 %d 条查询。
3. 对中文近期/新闻问题，可以补充英文关键词。
4. 不要编造实体，不要返回解释。`, r.maxQueries)),
		schema.UserMessage(fmt.Sprintf("用户问题：%s\n\n只返回 JSON 数组。", query)),
	})
	if err != nil {
		return nil, fmt.Errorf("generate rewritten search queries: %w", err)
	}
	if resp == nil || strings.TrimSpace(resp.Content) == "" {
		return nil, errors.New("query rewrite model returned empty content")
	}
	queries, err := parseQueryRewriteJSON(resp.Content)
	if err != nil {
		return nil, fmt.Errorf("parse rewritten search queries: %w", err)
	}
	return limitQueries(queries, r.maxQueries), nil
}

func parseQueryRewriteJSON(content string) ([]string, error) {
	content = strings.TrimSpace(content)
	var queries []string
	if err := json.Unmarshal([]byte(content), &queries); err == nil {
		return normalizeQueryList(queries), nil
	}
	start := strings.Index(content, "[")
	end := strings.LastIndex(content, "]")
	if start < 0 || end <= start {
		return nil, errors.New("no JSON array found")
	}
	if err := json.Unmarshal([]byte(content[start:end+1]), &queries); err != nil {
		return nil, err
	}
	return normalizeQueryList(queries), nil
}

func normalizeQueryList(queries []string) []string {
	out := make([]string, 0, len(queries))
	seen := map[string]struct{}{}
	for _, query := range queries {
		query = strings.TrimSpace(query)
		if query == "" {
			continue
		}
		key := strings.ToLower(query)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, query)
	}
	return out
}

func limitQueries(queries []string, maxQueries int) []string {
	if maxQueries <= 0 || len(queries) <= maxQueries {
		return queries
	}
	return append([]string(nil), queries[:maxQueries]...)
}
