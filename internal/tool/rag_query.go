package tool

import (
	"context"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
	"github.com/raiki02/video-extractor/internal/rag"
)

// RAGQueryInput is the input for the RAG retrieval tool.
type RAGQueryInput struct {
	Query string `json:"query" jsonschema:"required" jsonschema_description:"The user's question for retrieving relevant context."`
}

type RAGQueryOutput struct {
	Chunks []rag.RelevantChunk `json:"chunks"`
}

func NewRAGQueryTool(retriever *rag.Retriever) (tool.InvokableTool, *Wrapper, error) {
	inner, err := utils.InferTool(
		"rag_query",
		"Search the RAG knowledge base. Embeds the query, retrieves relevant chunks from Qdrant, and reranks them.",
		func(ctx context.Context, input RAGQueryInput) (RAGQueryOutput, error) {
			chunks, err := retriever.Retrieve(ctx, input.Query)
			if err != nil {
				return RAGQueryOutput{}, err
			}
			return RAGQueryOutput{Chunks: chunks}, nil
		},
	)
	if err != nil {
		return nil, nil, err
	}
	wrapper := NewWrapper(inner, WrapperConfig{Name: "rag_query", Timeout: 0})
	return inner, wrapper, nil
}
