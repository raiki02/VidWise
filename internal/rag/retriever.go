package rag

import (
	"context"
	"fmt"
	"log/slog"

	pb "github.com/qdrant/go-client/qdrant"
	"github.com/raiki02/video-extractor/internal/model"
	qdrantclient "github.com/raiki02/video-extractor/internal/storage/qdrant"
)

// RelevantChunk is a retrieved chunk with relevance score.
type RelevantChunk struct {
	Text  string  `json:"text"`
	Score float64 `json:"score"`
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
		searchTopK:   10,
		rerankTopK:   3,
	}
}

// Retrieve embeds a query, searches Qdrant, and reranks results.
func (r *Retriever) Retrieve(ctx context.Context, query, userID, sessionID string) ([]RelevantChunk, error) {
	// 1. Embed query
	queryVec, err := r.embedClient.EmbedSingle(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}

	// 2. Search Qdrant
	points, err := r.qdrantClient.Points.Search(ctx, &pb.SearchPoints{
		CollectionName: r.collection,
		Vector:         toFloat32Slice(queryVec),
		Limit:          uint64(r.searchTopK),
		WithPayload:    &pb.WithPayloadSelector{SelectorOptions: &pb.WithPayloadSelector_Enable{Enable: true}},
		Filter: &pb.Filter{
			Must: []*pb.Condition{
				{
					ConditionOneOf: &pb.Condition_Field{
						Field: &pb.FieldCondition{
							Key: qdrantclient.FieldUserID,
							Match: &pb.Match{
								MatchValue: &pb.Match_Keyword{Keyword: userID},
							},
						},
					},
				},
				{
					ConditionOneOf: &pb.Condition_Field{
						Field: &pb.FieldCondition{
							Key: qdrantclient.FieldSessionID,
							Match: &pb.Match{
								MatchValue: &pb.Match_Keyword{Keyword: sessionID},
							},
						},
					},
				},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("search qdrant: %w", err)
	}

	if len(points.Result) == 0 {
		slog.Info("rag.retriever.no_results")
		return nil, nil
	}

	slog.Info("rag.retriever.search_done", "results", len(points.Result))

	// 3. Extract texts and scores from Qdrant results
	type qdrantHit struct {
		text  string
		score float64
	}
	hits := make([]qdrantHit, 0, len(points.Result))
	for _, p := range points.Result {
		text := getPayloadString(p.Payload, qdrantclient.FieldText)
		if text == "" {
			continue
		}
		hits = append(hits, qdrantHit{text: text, score: float64(p.Score)})
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
			// Fall back to vector search results
		} else {
			result := make([]RelevantChunk, 0, r.rerankTopK)
			for i, rr := range reranked {
				if i >= r.rerankTopK {
					break
				}
				result = append(result, RelevantChunk{Text: rr.Text, Score: rr.Score})
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
		result = append(result, RelevantChunk{Text: h.text, Score: h.score})
	}
	return result, nil
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
