package tool

import (
	"context"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
	"github.com/raiki02/video-extractor/internal/rag"
)

// RAGIndexInput is the input for the RAG indexing tool.
type RAGIndexInput struct {
	Text string `json:"text" jsonschema:"required" jsonschema_description:"The text to chunk, embed, and index into the vector database."`
}

type RAGIndexOutput struct {
	ChunkCount int `json:"chunk_count"`
}

func NewRAGIndexTool(indexer *rag.Indexer) (tool.InvokableTool, *Wrapper, error) {
	inner, err := utils.InferTool(
		"rag_index",
		"Index text into the RAG knowledge base. Splits text into chunks, generates embeddings, and stores them in the Qdrant vector database.",
		func(ctx context.Context, input RAGIndexInput) (RAGIndexOutput, error) {
			count, err := indexer.IndexText(ctx, input.Text)
			if err != nil {
				return RAGIndexOutput{}, err
			}
			return RAGIndexOutput{ChunkCount: count}, nil
		},
	)
	if err != nil {
		return nil, nil, err
	}
	wrapper := NewWrapper(inner, WrapperConfig{Name: "rag_index", Timeout: 0})
	return inner, wrapper, nil
}
