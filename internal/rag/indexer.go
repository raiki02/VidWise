package rag

import (
	"context"
	"fmt"
	"log/slog"

	pb "github.com/qdrant/go-client/qdrant"
	"github.com/raiki02/video-extractor/internal/model"
	qdrantclient "github.com/raiki02/video-extractor/internal/storage/qdrant"
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
		chunkRunes:   512,
		overlapRunes: 64,
	}
}

// IndexText splits text into chunks, embeds each, and upserts to Qdrant.
// Returns the number of chunks indexed.
func (idx *Indexer) IndexText(ctx context.Context, text, taskID, userID, sessionID string) (int, error) {
	chunks := ChunkText(text, idx.chunkRunes, idx.overlapRunes)
	if len(chunks) == 0 {
		return 0, nil
	}

	slog.Info("rag.indexer.chunked", "chunks", len(chunks))

	// Prepare texts for batch embedding
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

	// Build Qdrant points
	points := make([]*pb.PointStruct, len(chunks))
	for i, chunk := range chunks {
		points[i] = &pb.PointStruct{
			Id: &pb.PointId{
				PointIdOptions: &pb.PointId_Uuid{Uuid: fmt.Sprintf("%s_%d", taskID, i)},
			},
			Vectors: &pb.Vectors{
				VectorsOptions: &pb.Vectors_Vector{Vector: &pb.Vector{Data: toFloat32Slice(embeddings[i])}},
			},
			Payload: map[string]*pb.Value{
				qdrantclient.FieldTaskID:    {Kind: &pb.Value_StringValue{StringValue: taskID}},
				qdrantclient.FieldUserID:    {Kind: &pb.Value_StringValue{StringValue: userID}},
				qdrantclient.FieldSessionID: {Kind: &pb.Value_StringValue{StringValue: sessionID}},
				qdrantclient.FieldChunkIdx:  {Kind: &pb.Value_IntegerValue{IntegerValue: int64(i)}},
				qdrantclient.FieldText:      {Kind: &pb.Value_StringValue{StringValue: chunk.Text}},
			},
		}
	}

	// Batch upsert
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
