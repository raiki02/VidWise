package tool

import (
	"context"
	"errors"
	"strings"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
	"github.com/raiki02/vidwise/internal/rag"
)

// RAGQueryInput is the input for the RAG retrieval tool.
type RAGQueryInput struct {
	Query      string   `json:"query" jsonschema:"required" jsonschema_description:"The user's question for retrieving relevant context."`
	UserID     string   `json:"user_id,omitempty" jsonschema_description:"User id that scopes retrieval to that user's indexed content. Required unless session_id is provided."`
	SessionID  string   `json:"session_id,omitempty" jsonschema_description:"Session id that scopes retrieval to one chat/session. Required unless user_id is provided."`
	TopK       int      `json:"top_k,omitempty" jsonschema_description:"Optional number of final chunks to return."`
	SearchTopK int      `json:"search_top_k,omitempty" jsonschema_description:"Optional number of vector search candidates before reranking."`
	MinScore   *float64 `json:"min_score,omitempty" jsonschema_description:"Optional minimum vector relevance score. 0 disables score filtering."`
}

type RAGQueryOutput struct {
	Status string              `json:"status"`
	Count  int                 `json:"count"`
	Chunks []rag.RelevantChunk `json:"chunks"`
}

func NewRAGQueryTool(retriever *rag.Retriever) (tool.InvokableTool, *Wrapper, error) {
	if retriever == nil {
		return nil, nil, errors.New("rag retriever is required")
	}

	inner, err := utils.InferTool(
		"rag_query",
		"Search the RAG knowledge base within an explicit user_id or session_id scope. Embeds the query, retrieves relevant chunks from Qdrant, and reranks them.",
		func(ctx context.Context, input RAGQueryInput) (RAGQueryOutput, error) {
			req, err := normalizeRAGQueryInput(input)
			if err != nil {
				return RAGQueryOutput{}, err
			}
			chunks, err := retriever.RetrieveWithOptions(ctx, req)
			if err != nil {
				return RAGQueryOutput{}, err
			}
			return newRAGQueryOutput(chunks), nil
		},
	)
	if err != nil {
		return nil, nil, err
	}
	wrapper := NewWrapper(inner, WrapperConfig{Name: "rag_query", Timeout: 0})
	return inner, wrapper, nil
}

func newRAGQueryOutput(chunks []rag.RelevantChunk) RAGQueryOutput {
	status := "no_results"
	if len(chunks) > 0 {
		status = "retrieved"
	}
	return RAGQueryOutput{
		Status: status,
		Count:  len(chunks),
		Chunks: chunks,
	}
}

func normalizeRAGQueryInput(input RAGQueryInput) (rag.RetrieveRequest, error) {
	query := strings.TrimSpace(input.Query)
	if query == "" {
		return rag.RetrieveRequest{}, errors.New("query is required")
	}
	filter, err := rag.NewRetrieveFilterWithPolicy(input.UserID, input.SessionID, rag.StrictScopePolicy())
	if err != nil {
		return rag.RetrieveRequest{}, err
	}
	return rag.RetrieveRequest{
		Query:      query,
		Filter:     filter,
		TopK:       input.TopK,
		SearchTopK: input.SearchTopK,
		MinScore:   input.MinScore,
	}, nil
}
