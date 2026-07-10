package tool

import (
	"context"
	"errors"
	"strings"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
	"github.com/raiki02/vidwise/internal/rag"
)

// RAGDeleteInput deletes previously indexed RAG sources by stable source_id.
type RAGDeleteInput struct {
	SourceID  string   `json:"source_id,omitempty" jsonschema_description:"Stable source_id returned by rag_index or /upload."`
	SourceIDs []string `json:"source_ids,omitempty" jsonschema_description:"Stable source_ids returned by rag_index or /upload."`
	UserID    string   `json:"user_id,omitempty" jsonschema_description:"User id that scopes deletion to that user's indexed content. Required unless session_id is provided."`
	SessionID string   `json:"session_id,omitempty" jsonschema_description:"Session id that scopes deletion to one chat/session. Required unless user_id is provided."`
}

type RAGDeleteOutput struct {
	SourceIDs []string `json:"source_ids"`
}

func NewRAGDeleteTool(indexer *rag.Indexer) (tool.InvokableTool, *Wrapper, error) {
	return NewRAGDeleteToolWithRegistry(indexer, nil)
}

func NewRAGDeleteToolWithRegistry(indexer *rag.Indexer, registry rag.SourceRegistry) (tool.InvokableTool, *Wrapper, error) {
	return NewRAGDeleteToolWithManager(rag.NewSourceManager(indexer, nil, registry))
}

func NewRAGDeleteToolWithManager(manager *rag.SourceManager) (tool.InvokableTool, *Wrapper, error) {
	if manager == nil || !manager.CanIndex() {
		return nil, nil, errors.New("rag source manager is required")
	}
	inner, err := utils.InferTool(
		"rag_delete",
		"Delete indexed RAG chunks by stable source_id within an explicit user_id or session_id scope. Use the source_ids returned by rag_index or /upload.",
		func(ctx context.Context, input RAGDeleteInput) (RAGDeleteOutput, error) {
			req, err := normalizeRAGDeleteInput(input)
			if err != nil {
				return RAGDeleteOutput{}, err
			}
			result, err := manager.DeleteSourcesWithOptions(ctx, req)
			if err != nil {
				return RAGDeleteOutput{}, err
			}
			return RAGDeleteOutput{SourceIDs: result.SourceIDs}, nil
		},
	)
	if err != nil {
		return nil, nil, err
	}
	wrapper := NewWrapper(inner, WrapperConfig{Name: "rag_delete", Timeout: 0})
	return inner, wrapper, nil
}

func normalizeRAGDeleteInput(input RAGDeleteInput) (rag.DeleteRequest, error) {
	sourceIDs := make([]string, 0, len(input.SourceIDs)+1)
	sourceIDs = append(sourceIDs, input.SourceID)
	sourceIDs = append(sourceIDs, input.SourceIDs...)

	out := make([]string, 0, len(sourceIDs))
	seen := map[string]bool{}
	for _, sourceID := range sourceIDs {
		sourceID = strings.TrimSpace(sourceID)
		if sourceID == "" || seen[sourceID] {
			continue
		}
		out = append(out, sourceID)
		seen[sourceID] = true
	}
	if len(out) == 0 {
		return rag.DeleteRequest{}, errors.New("source_id is required")
	}
	filter, err := rag.NewRetrieveFilterWithPolicy(input.UserID, input.SessionID, rag.StrictScopePolicy())
	if err != nil {
		return rag.DeleteRequest{}, err
	}
	return rag.DeleteRequest{
		SourceIDs: out,
		Filter:    filter,
	}, nil
}
