package rag

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	pb "github.com/qdrant/go-client/qdrant"
	"github.com/raiki02/vidwise/internal/model"
	qdrantclient "github.com/raiki02/vidwise/internal/storage/qdrant"
)

// RelevantChunk is a retrieved chunk with relevance score and metadata.
type RelevantChunk struct {
	Text  string  `json:"text"`
	Score float64 `json:"score"`
	// Source metadata for citation
	SourceID      string `json:"source_id,omitempty"`
	DocumentID    string `json:"document_id,omitempty"`
	ChunkID       string `json:"chunk_id,omitempty"`
	ContentHash   string `json:"content_hash,omitempty"`
	TaskID        string `json:"task_id,omitempty"`
	SessionID     string `json:"session_id,omitempty"`
	ChunkIdx      int64  `json:"chunk_idx,omitempty"`
	SourceName    string `json:"source_name,omitempty"`
	SourceURL     string `json:"source_url,omitempty"`
	ContentType   string `json:"content_type,omitempty"`
	DocumentTitle string `json:"document_title,omitempty"`
	HeadingPath   string `json:"heading_path,omitempty"`
}

// RetrieveFilter scopes retrieval to a specific user or session.
type RetrieveFilter struct {
	UserID      string
	SessionID   string
	SourceIDs   []string
	DocumentIDs []string
}

// NewRetrieveFilter creates a normalized retrieval scope using the default
// policy. A nil filter means the query intentionally targets the unscoped
// knowledge base.
func NewRetrieveFilter(userID, sessionID string) *RetrieveFilter {
	filter, _ := NewRetrieveFilterWithPolicy(userID, sessionID, DefaultScopePolicy())
	return filter
}

// RetrieveRequest is the single interface for RAG retrieval. It keeps query
// text, tenant/session scope, and retrieval sizing together so callers do not
// need to know Qdrant/rerank ordering details.
type RetrieveRequest struct {
	Query      string
	Filter     *RetrieveFilter
	SearchTopK int
	TopK       int
	MinScore   *float64
}

// RetrieverConfig controls default retrieval behaviour.
type RetrieverConfig struct {
	SearchTopK int
	TopK       int
	MinScore   float64
}

func DefaultRetrieverConfig() RetrieverConfig {
	return RetrieverConfig{
		SearchTopK: 20,
		TopK:       8,
		MinScore:   0,
	}
}

type retrievalHit struct {
	text          string
	score         float64
	sourceID      string
	documentID    string
	chunkID       string
	contentHash   string
	taskID        string
	sessionID     string
	chunkIdx      int64
	sourceName    string
	sourceURL     string
	contentType   string
	documentTitle string
	headingPath   string
}

type queryEmbedder interface {
	EmbedSingle(ctx context.Context, text string) ([]float64, error)
}

type vectorSearcher interface {
	Search(ctx context.Context, req *pb.SearchPoints) (*pb.SearchResponse, error)
}

type chunkReranker interface {
	Rerank(ctx context.Context, query string, documents []string) ([]model.RerankResult, error)
}

// Retriever handles query embedding, Qdrant search, and reranking.
type Retriever struct {
	embedder   queryEmbedder
	searcher   vectorSearcher
	reranker   chunkReranker
	collection string
	searchTopK int
	topK       int
	minScore   float64
}

func NewRetriever(embedClient *model.EmbedClient, rerankClient *model.RerankClient, qdrantClient *qdrantclient.Client, collection string) *Retriever {
	return NewRetrieverWithConfig(embedClient, rerankClient, qdrantClient, collection, DefaultRetrieverConfig())
}

func NewRetrieverWithConfig(embedClient *model.EmbedClient, rerankClient *model.RerankClient, qdrantClient *qdrantclient.Client, collection string, cfg RetrieverConfig) *Retriever {
	var searcher vectorSearcher
	if qdrantClient != nil {
		searcher = qdrantVectorSearcher{points: qdrantClient.Points}
	}
	return newRetrieverWithAdapters(embedClient, rerankClient, searcher, collection, cfg)
}

func newRetrieverWithAdapters(embedder queryEmbedder, reranker chunkReranker, searcher vectorSearcher, collection string, cfg RetrieverConfig) *Retriever {
	cfg = normalizeRetrieverConfig(cfg)
	return &Retriever{
		embedder:   embedder,
		searcher:   searcher,
		reranker:   reranker,
		collection: collection,
		searchTopK: cfg.SearchTopK,
		topK:       cfg.TopK,
		minScore:   cfg.MinScore,
	}
}

type qdrantVectorSearcher struct {
	points pb.PointsClient
}

func (s qdrantVectorSearcher) Search(ctx context.Context, req *pb.SearchPoints) (*pb.SearchResponse, error) {
	return s.points.Search(ctx, req)
}

// Retrieve embeds a query, searches Qdrant (scoped by filter if provided),
// and reranks results.
func (r *Retriever) Retrieve(ctx context.Context, query string, filter *RetrieveFilter) ([]RelevantChunk, error) {
	return r.RetrieveWithOptions(ctx, RetrieveRequest{Query: query, Filter: filter})
}

// RetrieveWithOptions embeds a query, searches Qdrant, and reranks results.
func (r *Retriever) RetrieveWithOptions(ctx context.Context, req RetrieveRequest) ([]RelevantChunk, error) {
	req = r.normalizeRetrieveRequest(req)
	if req.Query == "" {
		return nil, errors.New("rag query is required")
	}
	if r.embedder == nil {
		return nil, errors.New("rag embedder is required")
	}
	if r.searcher == nil {
		return nil, errors.New("rag vector searcher is required")
	}

	// 1. Embed query
	queryVec, err := r.embedder.EmbedSingle(ctx, req.Query)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}

	// 2. Build search request with optional filter
	searchReq := &pb.SearchPoints{
		CollectionName: r.collection,
		Vector:         toFloat32Slice(queryVec),
		Limit:          uint64(req.SearchTopK),
		WithPayload:    &pb.WithPayloadSelector{SelectorOptions: &pb.WithPayloadSelector_Enable{Enable: true}},
	}

	// Apply user/session filter for multi-tenant isolation.
	// When a user_id is provided, we scope to that user's indexed content.
	// When only a session_id is provided, we scope to that session.
	if req.Filter != nil {
		searchReq.Filter = buildScopeFilter(req.Filter)
	}

	points, err := r.searcher.Search(ctx, searchReq)
	if err != nil {
		return nil, fmt.Errorf("search qdrant: %w", err)
	}

	if len(points.Result) == 0 {
		slog.Info("rag.retriever.no_results")
		return nil, nil
	}

	slog.Info("rag.retriever.search_done", "results", len(points.Result))

	// 3. Extract texts, scores, and metadata from Qdrant results
	hits := make([]retrievalHit, 0, len(points.Result))
	for _, p := range points.Result {
		text := getPayloadString(p.Payload, qdrantclient.FieldText)
		if text == "" {
			continue
		}
		score := float64(p.Score)
		if !keepScore(score, *req.MinScore) {
			continue
		}
		hits = append(hits, retrievalHit{
			text:          text,
			score:         score,
			sourceID:      getPayloadString(p.Payload, qdrantclient.FieldSourceID),
			documentID:    getPayloadString(p.Payload, qdrantclient.FieldDocumentID),
			chunkID:       getPayloadString(p.Payload, qdrantclient.FieldChunkID),
			contentHash:   getPayloadString(p.Payload, qdrantclient.FieldContentHash),
			taskID:        getPayloadString(p.Payload, qdrantclient.FieldTaskID),
			sessionID:     getPayloadString(p.Payload, qdrantclient.FieldSessionID),
			chunkIdx:      getPayloadInt(p.Payload, qdrantclient.FieldChunkIdx),
			sourceName:    getPayloadString(p.Payload, qdrantclient.FieldSourceName),
			sourceURL:     getPayloadString(p.Payload, qdrantclient.FieldSourceURL),
			contentType:   getPayloadString(p.Payload, qdrantclient.FieldContentType),
			documentTitle: getPayloadString(p.Payload, qdrantclient.FieldDocumentTitle),
			headingPath:   getPayloadString(p.Payload, qdrantclient.FieldHeadingPath),
		})
	}

	// 4. Rerank
	if r.reranker != nil && len(hits) > 1 {
		docs := make([]string, len(hits))
		for i, h := range hits {
			docs[i] = h.text
		}
		reranked, err := r.reranker.Rerank(ctx, req.Query, docs)
		if err != nil {
			slog.Warn("rag.retriever.rerank_failed", "err", err)
		} else {
			result := make([]RelevantChunk, 0, req.TopK)
			for _, rr := range reranked {
				if len(result) >= req.TopK {
					break
				}
				chunk, ok := relevantChunkFromRerank(hits, rr)
				if !ok {
					slog.Warn("rag.retriever.rerank_invalid_index", "index", rr.Index, "hits", len(hits))
					continue
				}
				result = append(result, chunk)
			}
			return result, nil
		}
	}

	// Fallback: return top vector search results
	result := make([]RelevantChunk, 0, req.TopK)
	for i, h := range hits {
		if i >= req.TopK {
			break
		}
		result = append(result, relevantChunkFromHit(h, h.text, h.score))
	}
	return result, nil
}

func (r *Retriever) normalizeRetrieveRequest(req RetrieveRequest) RetrieveRequest {
	req.Query = strings.TrimSpace(req.Query)
	if req.SearchTopK <= 0 {
		req.SearchTopK = r.searchTopK
	}
	if req.TopK <= 0 {
		req.TopK = r.topK
	}
	if req.TopK > req.SearchTopK {
		req.TopK = req.SearchTopK
	}
	minScore := r.minScore
	if req.MinScore != nil {
		minScore = *req.MinScore
	}
	if minScore < 0 {
		minScore = 0
	}
	req.MinScore = &minScore
	if req.Filter != nil {
		req.Filter = NormalizeRetrieveFilter(req.Filter)
	}
	return req
}

func normalizeRetrieverConfig(cfg RetrieverConfig) RetrieverConfig {
	defaults := DefaultRetrieverConfig()
	if cfg.SearchTopK <= 0 {
		cfg.SearchTopK = defaults.SearchTopK
	}
	if cfg.TopK <= 0 {
		cfg.TopK = defaults.TopK
	}
	if cfg.TopK > cfg.SearchTopK {
		cfg.TopK = cfg.SearchTopK
	}
	if cfg.MinScore < 0 {
		cfg.MinScore = 0
	}
	return cfg
}

func keepScore(score, minScore float64) bool {
	return minScore <= 0 || score >= minScore
}

func relevantChunkFromRerank(hits []retrievalHit, rr model.RerankResult) (RelevantChunk, bool) {
	if rr.Index < 0 || rr.Index >= len(hits) {
		return RelevantChunk{}, false
	}
	hit := hits[rr.Index]
	text := strings.TrimSpace(rr.Text)
	if text == "" {
		text = hit.text
	}
	return relevantChunkFromHit(hit, text, rr.Score), true
}

func relevantChunkFromHit(hit retrievalHit, text string, score float64) RelevantChunk {
	return RelevantChunk{
		Text:          text,
		Score:         score,
		SourceID:      hit.sourceID,
		DocumentID:    hit.documentID,
		ChunkID:       hit.chunkID,
		ContentHash:   hit.contentHash,
		TaskID:        hit.taskID,
		SessionID:     hit.sessionID,
		ChunkIdx:      hit.chunkIdx,
		SourceName:    hit.sourceName,
		SourceURL:     hit.sourceURL,
		ContentType:   hit.contentType,
		DocumentTitle: hit.documentTitle,
		HeadingPath:   hit.headingPath,
	}
}

// buildScopeFilter creates a Qdrant filter from a RetrieveFilter.
func buildScopeFilter(f *RetrieveFilter) *pb.Filter {
	if f == nil {
		return nil
	}
	var mustClauses []*pb.Condition

	if f.UserID != "" {
		mustClauses = append(mustClauses, keywordCondition(qdrantclient.FieldUserID, f.UserID))
	}

	if f.SessionID != "" {
		mustClauses = append(mustClauses, keywordCondition(qdrantclient.FieldSessionID, f.SessionID))
	}

	if len(f.SourceIDs) > 0 {
		mustClauses = append(mustClauses, keywordsCondition(qdrantclient.FieldSourceID, f.SourceIDs))
	}

	if len(f.DocumentIDs) > 0 {
		mustClauses = append(mustClauses, keywordsCondition(qdrantclient.FieldDocumentID, f.DocumentIDs))
	}

	if len(mustClauses) == 0 {
		return nil
	}

	return &pb.Filter{
		Must: mustClauses,
	}
}

// NormalizeRetrieveFilter trims and deduplicates all retrieval filter fields.
func NormalizeRetrieveFilter(f *RetrieveFilter) *RetrieveFilter {
	if f == nil {
		return nil
	}
	normalized := &RetrieveFilter{
		UserID:      strings.TrimSpace(f.UserID),
		SessionID:   strings.TrimSpace(f.SessionID),
		SourceIDs:   normalizeFilterIDs(f.SourceIDs),
		DocumentIDs: normalizeFilterIDs(f.DocumentIDs),
	}
	if normalized.UserID == "" && normalized.SessionID == "" && len(normalized.SourceIDs) == 0 && len(normalized.DocumentIDs) == 0 {
		return nil
	}
	return normalized
}

func normalizeFilterIDs(ids []string) []string {
	if len(ids) == 0 {
		return nil
	}
	out := make([]string, 0, len(ids))
	seen := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		id = strings.TrimSpace(id)
		if id == "" {
			continue
		}
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		out = append(out, id)
	}
	return out
}

func keywordCondition(key, value string) *pb.Condition {
	return &pb.Condition{
		ConditionOneOf: &pb.Condition_Field{
			Field: &pb.FieldCondition{
				Key:   key,
				Match: &pb.Match{MatchValue: &pb.Match_Keyword{Keyword: value}},
			},
		},
	}
}

func keywordsCondition(key string, values []string) *pb.Condition {
	return &pb.Condition{
		ConditionOneOf: &pb.Condition_Field{
			Field: &pb.FieldCondition{
				Key: key,
				Match: &pb.Match{MatchValue: &pb.Match_Keywords{Keywords: &pb.RepeatedStrings{
					Strings: values,
				}}},
			},
		},
	}
}

func getPayloadString(payload map[string]*pb.Value, key string) string {
	if payload == nil {
		return ""
	}
	v, ok := payload[key]
	if !ok {
		return ""
	}
	if sv := v.GetStringValue(); sv != "" {
		return sv
	}
	return ""
}

func getPayloadInt(payload map[string]*pb.Value, key string) int64 {
	if payload == nil {
		return 0
	}
	v, ok := payload[key]
	if !ok {
		return 0
	}
	return v.GetIntegerValue()
}
