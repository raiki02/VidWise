package rag

import (
	"context"
	"fmt"
	"log/slog"
	"strconv"
	"strings"

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

	if err := qdrantclient.EnsureCollection(ctx, idx.qdrantClient, idx.collection, dim); err != nil {
		return err
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
	return idx.IndexDocumentsScoped(ctx, []Document{{
		PageContent: text,
		Metadata: map[string]string{
			qdrantclient.FieldContentType: PlainContentType,
		},
	}}, userID, sessionID)
}

// IndexMarkdown parses a Markdown document, enriches sections with heading
// metadata, chunks them, embeds the chunks, and stores them in Qdrant.
func (idx *Indexer) IndexMarkdown(ctx context.Context, markdownText string, metadata map[string]string) (int, error) {
	return idx.IndexMarkdownScoped(ctx, markdownText, metadata, "", "")
}

// IndexMarkdownScoped indexes Markdown with user/session metadata.
func (idx *Indexer) IndexMarkdownScoped(ctx context.Context, markdownText string, metadata map[string]string, userID, sessionID string) (int, error) {
	docs := ParseMarkdownDocuments(markdownText, metadata)
	return idx.IndexDocumentsScoped(ctx, docs, userID, sessionID)
}

// IndexDocuments indexes pre-parsed documents.
func (idx *Indexer) IndexDocuments(ctx context.Context, docs []Document) (int, error) {
	return idx.IndexDocumentsScoped(ctx, docs, "", "")
}

// IndexDocumentsScoped indexes documents with user and session metadata for
// multi-tenant isolation.
func (idx *Indexer) IndexDocumentsScoped(ctx context.Context, docs []Document, userID, sessionID string) (int, error) {
	// Ensure collection exists with correct vector dimension
	if err := idx.EnsureCollection(ctx); err != nil {
		return 0, fmt.Errorf("ensure collection: %w", err)
	}

	chunks := idx.chunkDocuments(docs)
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
		for key, value := range chunk.Metadata {
			if key == "" || value == "" {
				continue
			}
			payload[key] = metadataPayloadValue(key, value)
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

type documentChunk struct {
	Text     string
	Metadata map[string]string
}

func (idx *Indexer) chunkDocuments(docs []Document) []documentChunk {
	var out []documentChunk
	for _, doc := range docs {
		content := strings.TrimSpace(doc.PageContent)
		if content == "" {
			continue
		}
		chunks := ChunkText(content, idx.chunkRunes, idx.overlapRunes)
		for _, chunk := range chunks {
			meta := mergeMetadata(doc.Metadata, map[string]string{
				qdrantclient.FieldChunkSource:       string(chunk.Source),
				qdrantclient.FieldSectionChunkIndex: strconv.Itoa(chunk.Index),
			})
			out = append(out, documentChunk{
				Text:     chunk.Text,
				Metadata: meta,
			})
		}
	}
	return out
}

func metadataPayloadValue(key, value string) *pb.Value {
	if isIntegerMetadataField(key) {
		if parsed, err := strconv.ParseInt(value, 10, 64); err == nil {
			return &pb.Value{Kind: &pb.Value_IntegerValue{IntegerValue: parsed}}
		}
	}
	return &pb.Value{Kind: &pb.Value_StringValue{StringValue: value}}
}

func isIntegerMetadataField(key string) bool {
	switch key {
	case qdrantclient.FieldHeadingLevel,
		qdrantclient.FieldSectionIndex,
		qdrantclient.FieldSectionChunkIndex:
		return true
	default:
		return false
	}
}

func toFloat32Slice(f64 []float64) []float32 {
	f32 := make([]float32, len(f64))
	for i, v := range f64 {
		f32[i] = float32(v)
	}
	return f32
}
