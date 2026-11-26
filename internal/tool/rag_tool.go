package tool

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"
	"github.com/raiki02/vidwise/internal/rag"
	"github.com/raiki02/vidwise/internal/vectorstore"
)

// Embedder is the embedding service dependency of the RAG tools.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float64, error)
}

// VectorStore is the Qdrant dependency of the RAG tools.
type VectorStore interface {
	ReplaceSource(ctx context.Context, sourceID string, chunks []vectorstore.Chunk, vectors [][]float64) error
	Search(ctx context.Context, vector []float32, topK uint64) ([]vectorstore.Hit, error)
}

type IndexInput struct {
	TranscriptPath string `json:"transcript_path" jsonschema:"required" jsonschema_description:"Absolute path of the transcript .txt file, as returned by transcribe_audio."`
	SourceName     string `json:"source_name" jsonschema:"required" jsonschema_description:"Human readable name of the source, e.g. video title."`
	SourceURL      string `json:"source_url,omitempty" jsonschema_description:"Original video URL."`
}

type IndexOutput struct {
	SourceID string `json:"source_id" jsonschema_description:"Stable ID of the indexed source."`
	Chunks   int    `json:"chunks" jsonschema_description:"Number of chunks indexed."`
}

const embedBatchSize = 16

// NewRAGIndexTool returns the index_transcript eino tool: read the transcript,
// split it into chunks, embed them, and upsert into the vector store under a
// stable source ID.
func NewRAGIndexTool(workDir string, chunkRunes, overlapRunes int, embedder Embedder, store VectorStore) (tool.InvokableTool, error) {
	inner, err := utils.InferTool(
		"index_transcript",
		"Split a transcript file into chunks, embed them, and store them in the knowledge base so later questions can retrieve them.",
		func(ctx context.Context, input IndexInput) (IndexOutput, error) {
			if err := PathInWorkDir(workDir, input.TranscriptPath); err != nil {
				return IndexOutput{}, err
			}
			name := strings.TrimSpace(input.SourceName)
			if name == "" {
				return IndexOutput{}, fmt.Errorf("source_name is required")
			}
			data, err := os.ReadFile(filepath.Clean(input.TranscriptPath))
			if err != nil {
				return IndexOutput{}, fmt.Errorf("read transcript: %w", err)
			}
			text := string(data)
			if strings.TrimSpace(text) == "" {
				return IndexOutput{}, fmt.Errorf("transcript is empty")
			}
			chunks := rag.Split(text, chunkRunes, overlapRunes)
			if len(chunks) == 0 {
				return IndexOutput{}, fmt.Errorf("transcript produced no chunks")
			}

			// Embed in batches.
			vectors := make([][]float64, 0, len(chunks))
			for start := 0; start < len(chunks); start += embedBatchSize {
				end := start + embedBatchSize
				if end > len(chunks) {
					end = len(chunks)
				}
				batch, err := embedder.Embed(ctx, chunks[start:end])
				if err != nil {
					return IndexOutput{}, fmt.Errorf("embed chunks: %w", err)
				}
				if len(batch) != end-start {
					return IndexOutput{}, fmt.Errorf("embedding count mismatch: got %d, want %d", len(batch), end-start)
				}
				vectors = append(vectors, batch...)
			}

			identity := strings.TrimSpace(input.SourceURL)
			if identity == "" {
				identity = filepath.Clean(input.TranscriptPath)
			}
			sourceID := vectorstore.SourceID(identity)
			storeChunks := make([]vectorstore.Chunk, len(chunks))
			for i, c := range chunks {
				storeChunks[i] = vectorstore.Chunk{
					Text:       c,
					Index:      i,
					SourceID:   sourceID,
					SourceName: name,
					SourceURL:  strings.TrimSpace(input.SourceURL),
				}
			}
			start := time.Now()
			if err := store.ReplaceSource(ctx, sourceID, storeChunks, vectors); err != nil {
				return IndexOutput{}, fmt.Errorf("store chunks: %w", err)
			}
			slog.Info("tool.index_transcript.done", "source_id", sourceID, "chunks", len(chunks), "elapsed", time.Since(start))
			return IndexOutput{SourceID: sourceID, Chunks: len(chunks)}, nil
		},
	)
	if err != nil {
		return nil, err
	}
	return inner, nil
}

type QueryInput struct {
	Query string `json:"query" jsonschema:"required" jsonschema_description:"Natural language question about the indexed videos."`
	TopK  int    `json:"top_k,omitempty" jsonschema_description:"Max results to return (1-20). Default 5."`
}

type QueryHit struct {
	Text       string  `json:"text"`
	SourceName string  `json:"source_name"`
	SourceURL  string  `json:"source_url,omitempty"`
	Score      float64 `json:"score"`
}

type QueryOutput struct {
	Results []QueryHit `json:"results"`
}

// NewRAGQueryTool returns the search_knowledge eino tool: embed the query and
// retrieve the top-K most similar chunks from the vector store.
func NewRAGQueryTool(embedder Embedder, store VectorStore, defaultTopK int) (tool.InvokableTool, error) {
	inner, err := utils.InferTool(
		"search_knowledge",
		"Search the indexed video knowledge base for chunks relevant to a question. Use this before answering questions about previously processed videos.",
		func(ctx context.Context, input QueryInput) (QueryOutput, error) {
			query := strings.TrimSpace(input.Query)
			if query == "" {
				return QueryOutput{}, fmt.Errorf("query is required")
			}
			topK := input.TopK
			if topK <= 0 {
				topK = defaultTopK
			}
			if topK > 20 {
				topK = 20
			}
			queryVec, err := embedder.Embed(ctx, []string{query})
			if err != nil {
				return QueryOutput{}, fmt.Errorf("embed query: %w", err)
			}
			if len(queryVec) != 1 {
				return QueryOutput{}, fmt.Errorf("embedding query returned %d vectors", len(queryVec))
			}
			hits, err := store.Search(ctx, toFloat32(queryVec[0]), uint64(topK))
			if err != nil {
				return QueryOutput{}, fmt.Errorf("search knowledge: %w", err)
			}
			results := make([]QueryHit, 0, len(hits))
			for _, h := range hits {
				results = append(results, QueryHit{
					Text:       h.Text,
					SourceName: h.SourceName,
					SourceURL:  h.SourceURL,
					Score:      float64(h.Score),
				})
			}
			return QueryOutput{Results: results}, nil
		},
	)
	if err != nil {
		return nil, err
	}
	return inner, nil
}

func toFloat32(data []float64) []float32 {
	out := make([]float32, len(data))
	for i, v := range data {
		out[i] = float32(v)
	}
	return out
}
