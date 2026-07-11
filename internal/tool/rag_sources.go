package tool

import (
	"context"
	"errors"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
	"github.com/raiki02/vidwise/internal/rag"
)

type RAGListSourcesInput struct {
	UserID    string `json:"user_id,omitempty" jsonschema_description:"User id for the personal knowledge-base scope. Required unless session_id is provided."`
	SessionID string `json:"session_id,omitempty" jsonschema_description:"Fallback session scope used only when user_id is absent."`
	Limit     int    `json:"limit,omitempty" jsonschema_description:"Optional maximum number of sources to return."`
}

type RAGListSourcesOutput struct {
	Sources []rag.SourceSummary `json:"sources"`
}

func NewRAGListSourcesTool(catalog *rag.SourceCatalog) (tool.InvokableTool, *Wrapper, error) {
	return NewRAGListSourcesToolWithManager(rag.NewSourceManager(nil, catalog, nil))
}

func NewRAGListSourcesToolWithManager(manager *rag.SourceManager) (tool.InvokableTool, *Wrapper, error) {
	if manager == nil || !manager.CanList() {
		return nil, nil, errors.New("rag source manager is required")
	}

	inner, err := utils.InferTool(
		"rag_list_sources",
		"List indexed RAG sources. user_id is treated as the personal knowledge-base scope; session_id is used only when user_id is absent.",
		func(ctx context.Context, input RAGListSourcesInput) (RAGListSourcesOutput, error) {
			req, err := normalizeRAGListSourcesInput(input)
			if err != nil {
				return RAGListSourcesOutput{}, err
			}
			sources, err := manager.ListSources(ctx, req)
			if err != nil {
				return RAGListSourcesOutput{}, err
			}
			return RAGListSourcesOutput{Sources: sources}, nil
		},
	)
	if err != nil {
		return nil, nil, err
	}
	wrapper := NewWrapper(inner, WrapperConfig{Name: "rag_list_sources", Timeout: 0})
	return inner, wrapper, nil
}

func normalizeRAGListSourcesInput(input RAGListSourcesInput) (rag.SourceListRequest, error) {
	filter, err := rag.NewRetrieveFilterWithPolicy(input.UserID, input.SessionID, rag.PersonalKnowledgeScopePolicy())
	if err != nil {
		return rag.SourceListRequest{}, err
	}
	if input.Limit < 0 {
		return rag.SourceListRequest{}, errors.New("limit must be a positive integer")
	}
	return rag.SourceListRequest{
		Filter: filter,
		Limit:  input.Limit,
	}, nil
}
