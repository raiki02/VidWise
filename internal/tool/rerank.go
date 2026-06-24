package tool

import (
	"context"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
	"github.com/raiki02/video-extractor/internal/model"
)

// RerankInput is the input for the rerank tool.
type RerankInput struct {
	Query     string   `json:"query" jsonschema:"required" jsonschema_description:"The search query."`
	Documents []string `json:"documents" jsonschema:"required" jsonschema_description:"The list of candidate documents to rerank."`
}

type RerankOutput struct {
	Results []model.RerankResult `json:"results"`
}

func NewRerankTool(client *model.RerankClient) (tool.InvokableTool, *Wrapper, error) {
	inner, err := utils.InferTool(
		"rerank_documents",
		"Rerank a list of documents by relevance to a query using the configured reranking model.",
		func(ctx context.Context, input RerankInput) (RerankOutput, error) {
			results, err := client.Rerank(ctx, input.Query, input.Documents)
			if err != nil {
				return RerankOutput{}, err
			}
			return RerankOutput{Results: results}, nil
		},
	)
	if err != nil {
		return nil, nil, err
	}
	wrapper := NewWrapper(inner, WrapperConfig{Name: "rerank_documents", Timeout: 0})
	return inner, wrapper, nil
}
