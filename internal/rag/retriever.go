package rag

import (
	"context"
	"fmt"
	"log/slog"

	pb "github.com/qdrant/go-client/qdrant"
	"github.com/raiki02/vidwise/internal/model"
	qdrantclient "github.com/raiki02/vidwise/internal/storage/qdrant"
)

// RelevantChunk is a retrieved chunk with relevance score and metadata.
type RelevantChunk struct {
	Text  string  `json:"text"`
	Score float64 `json:"score"`
	// Source metadata for citation
	SessionID string `json:"session_id,omitempty"`
	ChunkIdx  int64  `json:"chunk_idx,omitempty"`
}

// RetrieveFilter scopes retrieval to a specific user or session.
type RetrieveFilter struct {
	UserID    string
	SessionID string
}

// Retriever handles query embedding, Qdrant search, and reranking.
type Retriever struct {
	embedClient  *model.EmbedClient
	rerankClient *model.RerankClient
	qdrantClient *qdrantclient.Client
	collection   string
	searchTopK   int
	rerankTopK   int
}

func NewRetriever(embedClient *model.EmbedClient, rerankClient *model.RerankClient, qdrantClient *qdrantclient.Client, collection string) *Retriever {
	return &Retriever{
		embedClient:  embedClient,
		rerankClient: rerankClient,
		qdrantClient: qdrantClient,
		collection:   collection,
		searchTopK:   20,
		rerankTopK:   8,
	}
}

// Retrieve embeds a query, searches Qdrant (scoped by filter if provided),
// and reranks results.
func (r *Retriever) Retrieve(ctx context.Context, query string, filter *RetrieveFilter) ([]RelevantChunk, error) {
	// 1. Embed query
	queryVec, err := r.embedClient.EmbedSingle(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}

	// 2. Build search request with optional filter
	searchReq := &pb.SearchPoints{
		CollectionName: r.collection,
		Vector:         toFloat32Slice(queryVec),
		Limit:          uint64(r.searchTopK),
		WithPayload:    &pb.WithPayloadSelector{SelectorOptions: &pb.WithPayloadSelector_Enable{Enable: true}},
	}

	// Apply user/session filter for multi-tenant isolation.
	// When a user_id is provided, we scope to that user's indexed content.
	// When only a session_id is provided, we scope to that session.
	if filter != nil {
		searchReq.Filter = buildScopeFilter(filter)
	}

	points, err := r.qdrantClient.Points.Search(ctx, searchReq)
	if err != nil {
		return nil, fmt.Errorf("search qdrant: %w", err)
	}

	if len(points.Result) == 0 {
		slog.Info("rag.retriever.no_results")
		return nil, nil
	}

	slog.Info("rag.retriever.search_done", "results", len(points.Result))

	// 3. Extract texts, scores, and metadata from Qdrant results
	type qdrantHit struct {
		text      string
		score     float64
		sessionID string
		chunkIdx  int64
	}
	hits := make([]qdrantHit, 0, len(points.Result))
	for _, p := range points.Result {
		text := getPayloadString(p.Payload, qdrantclient.FieldText)
		if text == "" {
			continue
		}
		hits = append(hits, qdrantHit{
			text:      text,
			score:     float64(p.Score),
			sessionID: getPayloadString(p.Payload, qdrantclient.FieldSessionID),
			chunkIdx:  getPayloadInt(p.Payload, qdrantclient.FieldChunkIdx),
		})
	}

	// 4. Rerank
	if r.rerankClient != nil && len(hits) > 1 {
		docs := make([]string, len(hits))
		for i, h := range hits {
			docs[i] = h.text
		}
		reranked, err := r.rerankClient.Rerank(ctx, query, docs)
		if err != nil {
			slog.Warn("rag.retriever.rerank_failed", "err", err)
		} else {
			result := make([]RelevantChunk, 0, r.rerankTopK)
			for i, rr := range reranked {
				if i >= r.rerankTopK {
					break
				}
				orig := hits[rr.Index]
				result = append(result, RelevantChunk{
					Text:      rr.Text,
					Score:     rr.Score,
					SessionID: orig.sessionID,
					ChunkIdx:  orig.chunkIdx,
				})
			}
			return result, nil
		}
	}

	// Fallback: return top vector search results
	result := make([]RelevantChunk, 0, r.rerankTopK)
	for i, h := range hits {
		if i >= r.rerankTopK {
			break
		}
		result = append(result, RelevantChunk{
			Text:      h.text,
			Score:     h.score,
			SessionID: h.sessionID,
			ChunkIdx:  h.chunkIdx,
		})
	}
	return result, nil
}

// buildScopeFilter creates a Qdrant filter from a RetrieveFilter.
func buildScopeFilter(f *RetrieveFilter) *pb.Filter {
	var mustClauses []*pb.Condition

	if f.UserID != "" {
		mustClauses = append(mustClauses, &pb.Condition{
			ConditionOneOf: &pb.Condition_Field{
				Field: &pb.FieldCondition{
					Key:   qdrantclient.FieldUserID,
					Match: &pb.Match{MatchValue: &pb.Match_Keyword{Keyword: f.UserID}},
				},
			},
		})
	}

	if f.SessionID != "" {
		mustClauses = append(mustClauses, &pb.Condition{
			ConditionOneOf: &pb.Condition_Field{
				Field: &pb.FieldCondition{
					Key:   qdrantclient.FieldSessionID,
					Match: &pb.Match{MatchValue: &pb.Match_Keyword{Keyword: f.SessionID}},
				},
			},
		})
	}

	if len(mustClauses) == 0 {
		return nil
	}

	return &pb.Filter{
		Must: mustClauses,
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
