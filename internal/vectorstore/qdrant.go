package vectorstore

import (
	"context"
	"crypto/sha256"
	"fmt"
	"log/slog"

	"github.com/google/uuid"
	pb "github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	FieldText       = "text"
	FieldChunkIdx   = "chunk_index"
	FieldSourceID   = "source_id"
	FieldSourceName = "source_name"
	FieldSourceURL  = "source_url"
)

type Store struct {
	conn        *grpc.ClientConn
	points      pb.PointsClient
	collections pb.CollectionsClient
	collection  string
}

// Chunk is a text fragment to be upserted with its own vector.
type Chunk struct {
	Text       string
	Index      int
	SourceID   string
	SourceName string
	SourceURL  string
}

// Hit is a search result.
type Hit struct {
	Chunk
	Score float32
}

// New dials Qdrant over gRPC (insecure, matching the local dev setup),
// creates the collection if missing, and ensures the source_id keyword index.
func New(ctx context.Context, addr, collection string, vectorDim uint64) (*Store, error) {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("qdrant dial: %w", err)
	}
	s := &Store{
		conn:        conn,
		points:      pb.NewPointsClient(conn),
		collections: pb.NewCollectionsClient(conn),
		collection:  collection,
	}
	if err := s.ensureCollection(ctx, vectorDim); err != nil {
		conn.Close()
		return nil, err
	}
	slog.Info("vectorstore.ready", "addr", addr, "collection", collection, "dim", vectorDim)
	return s, nil
}

func (s *Store) Close() error {
	return s.conn.Close()
}

func (s *Store) ensureCollection(ctx context.Context, dim uint64) error {
	list, err := s.collections.List(ctx, &pb.ListCollectionsRequest{})
	if err != nil {
		return fmt.Errorf("list collections: %w", err)
	}
	for _, col := range list.Collections {
		if col.Name == s.collection {
			return nil
		}
	}
	slog.Info("vectorstore.collection.create", "name", s.collection, "dim", dim)
	_, err = s.collections.Create(ctx, &pb.CreateCollection{
		CollectionName: s.collection,
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
		return fmt.Errorf("create collection %s: %w", s.collection, err)
	}
	return nil
}

// ReplaceSource deletes all points of sourceID and upserts the new chunks.
// A vector with a dimension that does not match the collection is rejected by
// Qdrant.
func (s *Store) ReplaceSource(ctx context.Context, sourceID string, chunks []Chunk, vectors [][]float64) error {
	if len(chunks) == 0 {
		// An empty source just removes previous points.
		_, err := s.points.Delete(ctx, &pb.DeletePoints{
			CollectionName: s.collection,
			Points: &pb.PointsSelector{
				PointsSelectorOneOf: &pb.PointsSelector_Filter{
					Filter: sourceFilter(sourceID),
				},
			},
			Wait: boolPtr(true),
		})
		if err != nil {
			return fmt.Errorf("delete source %s: %w", sourceID, err)
		}
		return nil
	}
	if len(vectors) != len(chunks) {
		return fmt.Errorf("vector count %d does not match chunk count %d", len(vectors), len(chunks))
	}

	points := make([]*pb.PointStruct, len(chunks))
	for i, chunk := range chunks {
		points[i] = &pb.PointStruct{
			Id:      pointID(sourceID, i),
			Vectors: vector(vectors[i]),
			Payload: map[string]*pb.Value{
				FieldText:       {Kind: &pb.Value_StringValue{StringValue: chunk.Text}},
				FieldChunkIdx:   {Kind: &pb.Value_IntegerValue{IntegerValue: int64(chunk.Index)}},
				FieldSourceID:   {Kind: &pb.Value_StringValue{StringValue: sourceID}},
				FieldSourceName: {Kind: &pb.Value_StringValue{StringValue: chunk.SourceName}},
				FieldSourceURL:  {Kind: &pb.Value_StringValue{StringValue: chunk.SourceURL}},
			},
		}
	}

	_, err := s.points.Upsert(ctx, &pb.UpsertPoints{
		CollectionName: s.collection,
		Points:         points,
		Wait:           boolPtr(true),
	})
	if err != nil {
		return fmt.Errorf("upsert to qdrant: %w", err)
	}
	return nil
}

// Search finds the topK most similar points to vector.
func (s *Store) Search(ctx context.Context, vector []float32, topK uint64) ([]Hit, error) {
	if topK == 0 {
		return nil, nil
	}
	resp, err := s.points.Search(ctx, &pb.SearchPoints{
		CollectionName: s.collection,
		Vector:         vector,
		Limit:          topK,
		WithPayload:    &pb.WithPayloadSelector{SelectorOptions: &pb.WithPayloadSelector_Enable{Enable: true}},
	})
	if err != nil {
		return nil, fmt.Errorf("search qdrant: %w", err)
	}

	hits := make([]Hit, 0, len(resp.Result))
	for _, p := range resp.Result {
		text := payloadString(p.Payload, FieldText)
		if text == "" {
			continue
		}
		hits = append(hits, Hit{
			Chunk: Chunk{
				Text:       text,
				Index:      int(payloadInt(p.Payload, FieldChunkIdx)),
				SourceID:   payloadString(p.Payload, FieldSourceID),
				SourceName: payloadString(p.Payload, FieldSourceName),
				SourceURL:  payloadString(p.Payload, FieldSourceURL),
			},
			Score: p.Score,
		})
	}
	return hits, nil
}

func sourceFilter(sourceID string) *pb.Filter {
	return &pb.Filter{
		Must: []*pb.Condition{
			{
				ConditionOneOf: &pb.Condition_Field{
					Field: &pb.FieldCondition{
						Key: FieldSourceID,
						Match: &pb.Match{
							MatchValue: &pb.Match_Keyword{Keyword: sourceID},
						},
					},
				},
			},
		},
	}
}

func boolPtr(v bool) *bool {
	return &v
}

func pointID(sourceID string, chunkIndex int) *pb.PointId {
	sum := sha256.Sum256([]byte(fmt.Sprintf("point\x00%s\x00%d", sourceID, chunkIndex)))
	id := uuid.UUID(sum[:16])
	return &pb.PointId{PointIdOptions: &pb.PointId_Uuid{Uuid: id.String()}}
}

func vector(data []float64) *pb.Vectors {
	return &pb.Vectors{VectorsOptions: &pb.Vectors_Vector{Vector: &pb.Vector{Data: toFloat32(data)}}}
}

func toFloat32(data []float64) []float32 {
	out := make([]float32, len(data))
	for i, v := range data {
		out[i] = float32(v)
	}
	return out
}

func payloadString(payload map[string]*pb.Value, key string) string {
	if payload == nil {
		return ""
	}
	value, ok := payload[key]
	if !ok {
		return ""
	}
	return value.GetStringValue()
}

func payloadInt(payload map[string]*pb.Value, key string) int64 {
	if payload == nil {
		return 0
	}
	value, ok := payload[key]
	if !ok {
		return 0
	}
	return value.GetIntegerValue()
}

// SourceID returns a stable ID for a source URL/path.
func SourceID(identity string) string {
	sum := sha256.Sum256([]byte("source\x00" + identity))
	return uuid.UUID(sum[:16]).String()
}
