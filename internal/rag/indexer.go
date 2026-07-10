package rag

import (
	"context"
	"crypto/sha256"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"sync"

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
	mu           sync.RWMutex
	chunkConfig  ChunkConfig
}

func NewIndexer(embedClient *model.EmbedClient, qdrantClient *qdrantclient.Client, collection string) *Indexer {
	return &Indexer{
		embedClient:  embedClient,
		qdrantClient: qdrantClient,
		collection:   collection,
		chunkConfig:  DefaultChunkConfig(),
	}
}

// SetChunkParams overrides the default chunking parameters.
func (idx *Indexer) SetChunkParams(chunkRunes, overlapRunes int) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if chunkRunes > 0 {
		idx.chunkConfig.MaxRunes = chunkRunes
	}
	if overlapRunes >= 0 {
		idx.chunkConfig.OverlapRunes = overlapRunes
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
			return qdrantclient.EnsurePayloadIndexes(ctx, idx.qdrantClient, idx.collection)
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

// IndexSource detects the source format, parses it into documents, chunks them,
// embeds the chunks, and stores them in Qdrant.
func (idx *Indexer) IndexSource(ctx context.Context, source Source, opts IndexOptions) (IndexResult, error) {
	return idx.IndexSourceScoped(ctx, source, opts, "", "")
}

// IndexSourceScoped indexes a source with user/session metadata.
func (idx *Indexer) IndexSourceScoped(ctx context.Context, source Source, opts IndexOptions, userID, sessionID string) (IndexResult, error) {
	docs, format := DocumentsFromSource(source)
	indexed, err := idx.indexDocumentsScoped(ctx, docs, userID, sessionID, idx.chunkConfigWithOptions(opts))
	if err != nil {
		return IndexResult{}, err
	}
	return IndexResult{
		ChunkCount:  indexed.count,
		ContentType: string(format),
		SourceIDs:   indexed.sourceIDs,
		Sources:     indexed.sources,
	}, nil
}

// IndexDocuments indexes pre-parsed documents.
func (idx *Indexer) IndexDocuments(ctx context.Context, docs []Document) (int, error) {
	return idx.IndexDocumentsScoped(ctx, docs, "", "")
}

// IndexDocumentsScoped indexes documents with user and session metadata for
// multi-tenant isolation.
func (idx *Indexer) IndexDocumentsScoped(ctx context.Context, docs []Document, userID, sessionID string) (int, error) {
	indexed, err := idx.indexDocumentsScoped(ctx, docs, userID, sessionID, idx.defaultChunkConfig())
	return indexed.count, err
}

// DeleteSource removes all indexed chunks for one stable source_id.
func (idx *Indexer) DeleteSource(ctx context.Context, sourceID string) (DeleteResult, error) {
	return idx.DeleteSources(ctx, []string{sourceID})
}

// DeleteSources removes all indexed chunks for each stable source_id. The
// operation waits for Qdrant to apply each deletion before returning.
func (idx *Indexer) DeleteSources(ctx context.Context, sourceIDs []string) (DeleteResult, error) {
	return idx.DeleteSourcesWithOptions(ctx, DeleteRequest{SourceIDs: sourceIDs})
}

// DeleteSourcesWithOptions removes indexed chunks for stable source_id values,
// optionally scoped by user/session metadata.
func (idx *Indexer) DeleteSourcesWithOptions(ctx context.Context, req DeleteRequest) (DeleteResult, error) {
	sourceIDs := normalizeSourceIDs(req.SourceIDs)
	if len(sourceIDs) == 0 {
		return DeleteResult{}, fmt.Errorf("source_id is required")
	}
	if idx == nil || idx.qdrantClient == nil || idx.qdrantClient.Points == nil {
		return DeleteResult{}, fmt.Errorf("qdrant points client is required")
	}
	if req.Filter != nil {
		req.Filter = NewRetrieveFilter(req.Filter.UserID, req.Filter.SessionID)
	}
	if err := idx.deleteSources(ctx, sourceIDs, req.Filter); err != nil {
		return DeleteResult{}, err
	}
	return DeleteResult{SourceIDs: sourceIDs}, nil
}

type indexedDocuments struct {
	count     int
	sourceIDs []string
	sources   []SourceSummary
}

func (idx *Indexer) indexDocumentsScoped(ctx context.Context, docs []Document, userID, sessionID string, cfg ChunkConfig) (indexedDocuments, error) {
	// Ensure collection exists with correct vector dimension
	if err := idx.EnsureCollection(ctx); err != nil {
		return indexedDocuments{}, fmt.Errorf("ensure collection: %w", err)
	}

	chunks := chunkDocumentsWithConfig(docs, cfg)
	if len(chunks) == 0 {
		return indexedDocuments{}, nil
	}

	slog.Info("rag.indexer.chunked", "chunks", len(chunks))

	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Text
	}

	embeddings, err := idx.embedClient.Embed(ctx, texts)
	if err != nil {
		return indexedDocuments{}, fmt.Errorf("embed chunks: %w", err)
	}

	if len(embeddings) != len(texts) {
		return indexedDocuments{}, fmt.Errorf("embedding count mismatch: got %d, want %d", len(embeddings), len(texts))
	}

	points := make([]*pb.PointStruct, len(chunks))
	identities := make([]chunkIdentity, len(chunks))
	for i, chunk := range chunks {
		identity := newChunkIdentity(chunk, userID, sessionID, i)
		identities[i] = identity
		payload := map[string]*pb.Value{
			qdrantclient.FieldChunkIdx:    {Kind: &pb.Value_IntegerValue{IntegerValue: int64(i)}},
			qdrantclient.FieldText:        {Kind: &pb.Value_StringValue{StringValue: chunk.Text}},
			qdrantclient.FieldSourceID:    {Kind: &pb.Value_StringValue{StringValue: identity.sourceID}},
			qdrantclient.FieldDocumentID:  {Kind: &pb.Value_StringValue{StringValue: identity.documentID}},
			qdrantclient.FieldContentHash: {Kind: &pb.Value_StringValue{StringValue: identity.contentHash}},
			qdrantclient.FieldChunkID:     {Kind: &pb.Value_StringValue{StringValue: identity.chunkID}},
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
				PointIdOptions: &pb.PointId_Uuid{Uuid: stablePointUUID(identity.chunkID)},
			},
			Vectors: &pb.Vectors{
				VectorsOptions: &pb.Vectors_Vector{Vector: &pb.Vector{Data: toFloat32Slice(embeddings[i])}},
			},
			Payload: payload,
		}
	}

	sourceIDs := sourceIDsFromIdentities(identities)
	sources := sourceSummariesFromChunks(chunks, identities, userID, sessionID)
	if err := idx.deleteSources(ctx, sourceIDs, nil); err != nil {
		return indexedDocuments{}, err
	}

	_, err = idx.qdrantClient.Points.Upsert(ctx, &pb.UpsertPoints{
		CollectionName: idx.collection,
		Points:         points,
	})
	if err != nil {
		return indexedDocuments{}, fmt.Errorf("upsert to qdrant: %w", err)
	}

	slog.Info("rag.indexer.done", "chunks", len(chunks))
	return indexedDocuments{count: len(chunks), sourceIDs: sourceIDs, sources: sources}, nil
}

type chunkIdentity struct {
	sourceID    string
	documentID  string
	contentHash string
	chunkID     string
}

type documentChunk struct {
	Text     string
	Metadata map[string]string
}

func (idx *Indexer) defaultChunkConfig() ChunkConfig {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.chunkConfig
}

func (idx *Indexer) chunkConfigWithOptions(opts IndexOptions) ChunkConfig {
	cfg := idx.defaultChunkConfig()
	if opts.ChunkRunes > 0 {
		cfg.MaxRunes = opts.ChunkRunes
	}
	if opts.OverlapRunes != nil {
		cfg.OverlapRunes = *opts.OverlapRunes
	}
	return cfg
}

func (idx *Indexer) chunkDocuments(docs []Document) []documentChunk {
	return chunkDocumentsWithConfig(docs, idx.defaultChunkConfig())
}

func chunkDocumentsWithConfig(docs []Document, cfg ChunkConfig) []documentChunk {
	var out []documentChunk
	for _, doc := range docs {
		content := strings.TrimSpace(doc.PageContent)
		if content == "" {
			continue
		}
		chunks := chunkDocumentTextWithConfig(content, doc.Metadata, cfg)
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

func newChunkIdentity(chunk documentChunk, userID, sessionID string, chunkIndex int) chunkIdentity {
	contentHash := stableHash("content", normalizeFingerprintText(chunk.Text))
	sourceKey := stableSourceKey(chunk.Metadata, userID, sessionID, contentHash)
	sourceID := stableHash("source", sourceKey)
	documentKey := stableDocumentKey(chunk.Metadata, userID, sessionID, contentHash)
	documentID := stableHash("document", documentKey)
	chunkKey := strings.Join([]string{
		"chunk",
		documentID,
		strconv.Itoa(chunkIndex),
		chunk.Metadata[qdrantclient.FieldSectionChunkIndex],
		chunk.Metadata[qdrantclient.FieldChunkSource],
	}, "\x00")
	return chunkIdentity{
		sourceID:    sourceID,
		documentID:  documentID,
		contentHash: contentHash,
		chunkID:     stableHash("chunk", chunkKey),
	}
}

func sourceIDsFromIdentities(identities []chunkIdentity) []string {
	sourceIDs := make([]string, 0, 1)
	for _, identity := range identities {
		sourceIDs = append(sourceIDs, identity.sourceID)
	}
	return normalizeSourceIDs(sourceIDs)
}

func sourceSummariesFromChunks(chunks []documentChunk, identities []chunkIdentity, userID, sessionID string) []SourceSummary {
	bySource := map[string]*SourceSummary{}
	order := make([]string, 0, len(identities))
	for i, identity := range identities {
		if identity.sourceID == "" || i >= len(chunks) {
			continue
		}
		source := bySource[identity.sourceID]
		if source == nil {
			source = &SourceSummary{
				SourceID:  identity.sourceID,
				UserID:    strings.TrimSpace(userID),
				SessionID: strings.TrimSpace(sessionID),
			}
			bySource[identity.sourceID] = source
			order = append(order, identity.sourceID)
		}
		source.ChunkCount++
		source.SourceName = firstNonEmpty(source.SourceName, chunks[i].Metadata[qdrantclient.FieldSourceName])
		source.SourceURL = firstNonEmpty(source.SourceURL, chunks[i].Metadata[qdrantclient.FieldSourceURL])
		source.ContentType = firstNonEmpty(source.ContentType, chunks[i].Metadata[qdrantclient.FieldContentType])
		source.DocumentTitle = firstNonEmpty(source.DocumentTitle, chunks[i].Metadata[qdrantclient.FieldDocumentTitle])
		source.DocumentIDs = appendUniqueTrimmed(source.DocumentIDs, firstNonEmpty(identity.documentID, chunks[i].Metadata[qdrantclient.FieldDocumentID]), 0)
		source.TaskIDs = appendUniqueTrimmed(source.TaskIDs, chunks[i].Metadata[qdrantclient.FieldTaskID], 0)
		source.HeadingPaths = appendUniqueTrimmed(source.HeadingPaths, chunks[i].Metadata[qdrantclient.FieldHeadingPath], 12)
	}

	out := make([]SourceSummary, 0, len(order))
	for _, sourceID := range order {
		if source := bySource[sourceID]; source != nil {
			out = append(out, *source)
		}
	}
	return out
}

func normalizeSourceIDs(sourceIDs []string) []string {
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
	return out
}

func (idx *Indexer) deleteSources(ctx context.Context, sourceIDs []string, filter *RetrieveFilter) error {
	for _, sourceID := range sourceIDs {
		sourceID = strings.TrimSpace(sourceID)
		if sourceID == "" {
			continue
		}
		if _, err := idx.qdrantClient.Points.Delete(ctx, buildSourceDeleteRequest(idx.collection, sourceID, filter)); err != nil {
			return fmt.Errorf("delete existing source chunks %s: %w", sourceID, err)
		}
	}
	return nil
}

func buildSourceDeleteRequest(collection, sourceID string, filter *RetrieveFilter) *pb.DeletePoints {
	wait := true
	must := []*pb.Condition{
		keywordCondition(qdrantclient.FieldSourceID, sourceID),
	}
	if scopeFilter := buildScopeFilter(filter); scopeFilter != nil {
		must = append(must, scopeFilter.Must...)
	}
	return &pb.DeletePoints{
		CollectionName: collection,
		Wait:           &wait,
		Points: &pb.PointsSelector{
			PointsSelectorOneOf: &pb.PointsSelector_Filter{
				Filter: &pb.Filter{
					Must: must,
				},
			},
		},
	}
}

func stableSourceKey(metadata map[string]string, userID, sessionID, fallbackContentHash string) string {
	fallback := ""
	if !sourceIdentityHasStableMetadata(metadata) {
		fallback = fallbackContentHash
	}
	parts := []string{
		"user", strings.TrimSpace(userID),
		"session", strings.TrimSpace(sessionID),
		"source_name", strings.TrimSpace(metadata[qdrantclient.FieldSourceName]),
		"source_url", strings.TrimSpace(metadata[qdrantclient.FieldSourceURL]),
		"task_id", strings.TrimSpace(metadata[qdrantclient.FieldTaskID]),
		"content_type", strings.TrimSpace(metadata[qdrantclient.FieldContentType]),
		"fallback_content", fallback,
	}
	return strings.Join(parts, "\x00")
}

func sourceIdentityHasStableMetadata(metadata map[string]string) bool {
	sourceFields := []string{
		strings.TrimSpace(metadata[qdrantclient.FieldSourceName]),
		strings.TrimSpace(metadata[qdrantclient.FieldSourceURL]),
		strings.TrimSpace(metadata[qdrantclient.FieldTaskID]),
	}
	for _, field := range sourceFields {
		if field != "" {
			return true
		}
	}
	return false
}

func stableDocumentKey(metadata map[string]string, userID, sessionID, fallbackContentHash string) string {
	sourceFields := []string{
		strings.TrimSpace(metadata[qdrantclient.FieldSourceName]),
		strings.TrimSpace(metadata[qdrantclient.FieldSourceURL]),
		strings.TrimSpace(metadata[qdrantclient.FieldTaskID]),
		strings.TrimSpace(metadata[qdrantclient.FieldDocumentTitle]),
		strings.TrimSpace(metadata[qdrantclient.FieldHeadingPath]),
		strings.TrimSpace(metadata[qdrantclient.FieldSectionIndex]),
	}
	hasSourceIdentity := false
	for _, field := range sourceFields {
		if field != "" {
			hasSourceIdentity = true
			break
		}
	}
	fallback := ""
	if !hasSourceIdentity {
		fallback = fallbackContentHash
	}
	parts := []string{
		"user", strings.TrimSpace(userID),
		"session", strings.TrimSpace(sessionID),
		"source_name", strings.TrimSpace(metadata[qdrantclient.FieldSourceName]),
		"source_url", strings.TrimSpace(metadata[qdrantclient.FieldSourceURL]),
		"task_id", strings.TrimSpace(metadata[qdrantclient.FieldTaskID]),
		"content_type", strings.TrimSpace(metadata[qdrantclient.FieldContentType]),
		"document_title", strings.TrimSpace(metadata[qdrantclient.FieldDocumentTitle]),
		"heading_path", strings.TrimSpace(metadata[qdrantclient.FieldHeadingPath]),
		"section_index", strings.TrimSpace(metadata[qdrantclient.FieldSectionIndex]),
		"fallback_content", fallback,
	}
	return strings.Join(parts, "\x00")
}

func normalizeFingerprintText(text string) string {
	return strings.TrimSpace(strings.ReplaceAll(text, "\r\n", "\n"))
}

func stableHash(prefix, value string) string {
	sum := sha256.Sum256([]byte(prefix + "\x00" + value))
	return fmt.Sprintf("%x", sum[:])
}

func stablePointUUID(chunkID string) string {
	sum := sha256.Sum256([]byte("point\x00" + chunkID))
	var id uuid.UUID
	copy(id[:], sum[:16])
	return id.String()
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
