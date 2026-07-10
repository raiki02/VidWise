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
	Query        string   `json:"query" jsonschema:"required" jsonschema_description:"The user's question for retrieving relevant context."`
	UserID       string   `json:"user_id,omitempty" jsonschema_description:"User id that scopes retrieval to that user's indexed content. Required unless session_id is provided."`
	SessionID    string   `json:"session_id,omitempty" jsonschema_description:"Session id that scopes retrieval to one chat/session. Required unless user_id is provided."`
	SourceIDs    []string `json:"source_ids,omitempty" jsonschema_description:"Optional stable source_ids to limit retrieval within the scoped knowledge base."`
	DocumentIDs  []string `json:"document_ids,omitempty" jsonschema_description:"Optional document_ids to limit retrieval within the scoped knowledge base."`
	TopK         int      `json:"top_k,omitempty" jsonschema_description:"Optional number of final chunks to return."`
	SearchTopK   int      `json:"search_top_k,omitempty" jsonschema_description:"Optional number of vector search candidates before reranking."`
	MinScore     *float64 `json:"min_score,omitempty" jsonschema_description:"Optional minimum vector relevance score. 0 disables score filtering."`
	ContextRunes int      `json:"context_max_runes,omitempty" jsonschema_description:"Optional maximum runes for the prompt-ready packed context returned by this tool."`
}

type RAGQueryOutput struct {
	Status                   string                    `json:"status"`
	Count                    int                       `json:"count"`
	Chunks                   []rag.RelevantChunk       `json:"chunks"`
	Context                  string                    `json:"context,omitempty"`
	ContextUsedChunks        int                       `json:"context_used_chunks"`
	ContextSkippedDuplicates int                       `json:"context_skipped_duplicates"`
	ContextTruncated         bool                      `json:"context_truncated"`
	ContextCitations         []RAGQueryContextCitation `json:"context_citations,omitempty"`
}

type RAGQueryContextCitation struct {
	SnippetNumber int               `json:"snippet_number"`
	Chunk         rag.RelevantChunk `json:"chunk"`
}

type ragQueryRequest struct {
	Retrieve rag.RetrieveRequest
	Context  rag.ContextConfig
}

func NewRAGQueryTool(retriever *rag.Retriever) (tool.InvokableTool, *Wrapper, error) {
	if retriever == nil {
		return nil, nil, errors.New("rag retriever is required")
	}

	inner, err := utils.InferTool(
		"rag_query",
		"Search the RAG knowledge base within an explicit user_id or session_id scope. Embeds the query, retrieves relevant chunks from Qdrant, reranks them, and returns both raw chunks and prompt-ready packed context.",
		func(ctx context.Context, input RAGQueryInput) (RAGQueryOutput, error) {
			req, err := normalizeRAGQueryInput(input)
			if err != nil {
				return RAGQueryOutput{}, err
			}
			chunks, err := retriever.RetrieveWithOptions(ctx, req.Retrieve)
			if err != nil {
				return RAGQueryOutput{}, err
			}
			return newRAGQueryOutput(chunks, req.Context), nil
		},
	)
	if err != nil {
		return nil, nil, err
	}
	wrapper := NewWrapper(inner, WrapperConfig{Name: "rag_query", Timeout: 0})
	return inner, wrapper, nil
}

func newRAGQueryOutput(chunks []rag.RelevantChunk, contextCfg rag.ContextConfig) RAGQueryOutput {
	status := "no_results"
	if len(chunks) > 0 {
		status = "retrieved"
	}
	packedContext := rag.PackContext(chunks, contextCfg)
	return RAGQueryOutput{
		Status:                   status,
		Count:                    len(chunks),
		Chunks:                   chunks,
		Context:                  packedContext.Text,
		ContextUsedChunks:        packedContext.UsedChunks,
		ContextSkippedDuplicates: packedContext.SkippedDuplicates,
		ContextTruncated:         packedContext.Truncated,
		ContextCitations:         newRAGQueryContextCitations(packedContext.Citations),
	}
}

func newRAGQueryContextCitations(citations []rag.ContextCitation) []RAGQueryContextCitation {
	if len(citations) == 0 {
		return nil
	}
	out := make([]RAGQueryContextCitation, 0, len(citations))
	for _, citation := range citations {
		out = append(out, RAGQueryContextCitation{
			SnippetNumber: citation.SnippetNumber,
			Chunk:         citation.Chunk,
		})
	}
	return out
}

func normalizeRAGQueryInput(input RAGQueryInput) (ragQueryRequest, error) {
	query := strings.TrimSpace(input.Query)
	if query == "" {
		return ragQueryRequest{}, errors.New("query is required")
	}
	filter, err := rag.NewRetrieveFilterWithPolicy(input.UserID, input.SessionID, rag.StrictScopePolicy())
	if err != nil {
		return ragQueryRequest{}, err
	}
	filter.SourceIDs = input.SourceIDs
	filter.DocumentIDs = input.DocumentIDs
	filter = rag.NormalizeRetrieveFilter(filter)
	return ragQueryRequest{
		Retrieve: rag.RetrieveRequest{
			Query:      query,
			Filter:     filter,
			TopK:       input.TopK,
			SearchTopK: input.SearchTopK,
			MinScore:   input.MinScore,
		},
		Context: rag.ContextConfig{MaxRunes: input.ContextRunes},
	}, nil
}
