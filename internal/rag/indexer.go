package rag

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/google/uuid"
	pb "github.com/qdrant/go-client/qdrant"
	"github.com/raiki02/vidwise/internal/model"
	qdrantclient "github.com/raiki02/vidwise/internal/storage/qdrant"
)

// Indexer indexes text chunks into Qdrant.
type Indexer struct {
	embedClient  *model.EmbedClient
	qdrantClient *qdrantclient.Client
	collection   string
	chunkRunes   int
	overlapRunes int
}

func NewIndexer(embedClient *model.EmbedClient, qdrantClient *qdrantclient.Client, collection string) *Indexer {
	return &Indexer{
		embedClient:  embedClient,
		qdrantClient: qdrantClient,
		collection:   collection,
		chunkRunes:   1024,
		overlapRunes: 128,
	}
}

// SetChunkParams overrides the default chunking parameters.
func (idx *Indexer) SetChunkParams(chunkRunes, overlapRunes int) {
	if chunkRunes > 0 {
		idx.chunkRunes = chunkRunes
	}
	if overlapRunes >= 0 {
		idx.overlapRunes = overlapRunes
	}
}

// EnsureCollection checks if the collection exists and creates it with the
// correct vector dimension if not. The dimension is detected from a sample embedding.
func (idx *Indexer) EnsureCollection(ctx context.Context) error {
	listResp, err := idx.qdrantClient.Collections.List(ctx, &pb.ListCollectionsRequest{})
	if err != nil {
		return fmt.Errorf("qdrant list collections: %w", err)
	}
	for _, col := range listResp.Collections {
		if col.Name == idx.collection {
			slog.Info("rag.indexer.collection_exists", "name", idx.collection)
			return nil
		}
	}

	// Collection doesn't exist — detect dimension from a sample embedding
	slog.Info("rag.indexer.creating_collection", "name", idx.collection, "reason", "not found")
	dim, err := idx.detectDimension(ctx)
	if err != nil {
		return fmt.Errorf("detect embedding dimension: %w", err)
	}
	slog.Info("rag.indexer.detected_dim", "dim", dim)

	_, err = idx.qdrantClient.Collections.Create(ctx, &pb.CreateCollection{
		CollectionName: idx.collection,
		VectorsConfig: &pb.VectorsConfig{
			Config: &pb.VectorsConfig_Params{
				Params: &pb.VectorParams{
					Size:     dim,
					Distance: pb.Distance_Cosine,
				},
			},
		},
	})
	if err != nil {
		return fmt.Errorf("create collection %s: %w", idx.collection, err)
	}

	slog.Info("rag.indexer.collection_created", "name", idx.collection, "dim", dim)
	return nil
}

// detectDimension gets vector dimension from a sample embedding.
func (idx *Indexer) detectDimension(ctx context.Context) (uint64, error) {
	embeddings, err := idx.embedClient.Embed(ctx, []string{"dimension check"})
	if err != nil {
		return 0, fmt.Errorf("sample embed: %w", err)
	}
	if len(embeddings) == 0 || len(embeddings[0]) == 0 {
		return 0, fmt.Errorf("empty embedding returned")
	}
	return uint64(len(embeddings[0])), nil
}

// IndexText splits text into chunks, embeds each, and upserts to Qdrant.
// Automatically ensures the collection exists first.
// If userID/sessionID are provided, they are stored in the payload for later filtering.
func (idx *Indexer) IndexText(ctx context.Context, text string) (int, error) {
	return idx.IndexTextScoped(ctx, text, "", "")
}

// IndexTextScoped indexes text with user and session metadata for multi-tenant isolation.
func (idx *Indexer) IndexTextScoped(ctx context.Context, text, userID, sessionID string) (int, error) {
	// Ensure collection exists with correct vector dimension
	if err := idx.EnsureCollection(ctx); err != nil {
		return 0, fmt.Errorf("ensure collection: %w", err)
	}

	chunks := ChunkText(text, idx.chunkRunes, idx.overlapRunes)
	if len(chunks) == 0 {
		return 0, nil
	}

	slog.Info("rag.indexer.chunked", "chunks", len(chunks))

	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Text
	}

	embeddings, err := idx.embedClient.Embed(ctx, texts)
	if err != nil {
		return 0, fmt.Errorf("embed chunks: %w", err)
	}

	if len(embeddings) != len(texts) {
		return 0, fmt.Errorf("embedding count mismatch: got %d, want %d", len(embeddings), len(texts))
	}

	points := make([]*pb.PointStruct, len(chunks))
	for i, chunk := range chunks {
		payload := map[string]*pb.Value{
			qdrantclient.FieldChunkIdx: {Kind: &pb.Value_IntegerValue{IntegerValue: int64(i)}},
			qdrantclient.FieldText:     {Kind: &pb.Value_StringValue{StringValue: chunk.Text}},
		}
		if userID != "" {
			payload[qdrantclient.FieldUserID] = &pb.Value{Kind: &pb.Value_StringValue{StringValue: userID}}
		}
		if sessionID != "" {
			payload[qdrantclient.FieldSessionID] = &pb.Value{Kind: &pb.Value_StringValue{StringValue: sessionID}}
		}
		points[i] = &pb.PointStruct{
			Id: &pb.PointId{
				PointIdOptions: &pb.PointId_Uuid{Uuid: uuid.New().String()},
			},
			Vectors: &pb.Vectors{
				VectorsOptions: &pb.Vectors_Vector{Vector: &pb.Vector{Data: toFloat32Slice(embeddings[i])}},
			},
			Payload: payload,
		}
	}

	_, err = idx.qdrantClient.Points.Upsert(ctx, &pb.UpsertPoints{
		CollectionName: idx.collection,
		Points:         points,
	})
	if err != nil {
		return 0, fmt.Errorf("upsert to qdrant: %w", err)
	}

	slog.Info("rag.indexer.done", "chunks", len(chunks))
	return len(chunks), nil
}

func toFloat32Slice(f64 []float64) []float32 {
	f32 := make([]float32, len(f64))
	for i, v := range f64 {
		f32[i] = float32(v)
	}
	return f32
}
