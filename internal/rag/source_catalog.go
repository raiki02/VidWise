package rag

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"

	pb "github.com/qdrant/go-client/qdrant"
	qdrantclient "github.com/raiki02/vidwise/internal/storage/qdrant"
	"google.golang.org/grpc"
)

const (
	defaultSourceCatalogLimit      = 50
	defaultSourceCatalogScrollPage = 256
)

// SourceSummary is the operational view of one indexed RAG source.
type SourceSummary struct {
	SourceID      string `json:"source_id"`
	SourceName    string `json:"source_name,omitempty"`
	SourceURL     string `json:"source_url,omitempty"`
	ContentType   string `json:"content_type,omitempty"`
	DocumentTitle string `json:"document_title,omitempty"`
	UserID        string `json:"user_id,omitempty"`
	SessionID     string `json:"session_id,omitempty"`
	ChunkCount    int    `json:"chunk_count"`
}

// SourceListRequest describes source catalog listing. Filter is required by
// user-facing callers and may be nil only for internal/admin catalog access.
type SourceListRequest struct {
	Filter *RetrieveFilter
	Limit  int
}

type sourceScroller interface {
	Scroll(ctx context.Context, req *pb.ScrollPoints, opts ...grpc.CallOption) (*pb.ScrollResponse, error)
}

// SourceCatalog lists indexed RAG sources from vector payload metadata.
type SourceCatalog struct {
	scroller   sourceScroller
	registry   SourceRegistry
	collection string
	pageSize   uint32
}

func NewSourceCatalog(qdrantClient *qdrantclient.Client, collection string) *SourceCatalog {
	return NewSourceCatalogWithRegistry(qdrantClient, collection, nil)
}

func NewSourceCatalogWithRegistry(qdrantClient *qdrantclient.Client, collection string, registry SourceRegistry) *SourceCatalog {
	var scroller sourceScroller
	if qdrantClient != nil {
		scroller = qdrantClient.Points
	}
	return newSourceCatalogWithAdapters(scroller, registry, collection, defaultSourceCatalogScrollPage)
}

func newSourceCatalogWithAdapters(scroller sourceScroller, registry SourceRegistry, collection string, pageSize uint32) *SourceCatalog {
	if pageSize == 0 {
		pageSize = defaultSourceCatalogScrollPage
	}
	return &SourceCatalog{
		scroller:   scroller,
		registry:   registry,
		collection: collection,
		pageSize:   pageSize,
	}
}

func (c *SourceCatalog) ListSources(ctx context.Context, req SourceListRequest) ([]SourceSummary, error) {
	if c == nil || (c.scroller == nil && c.registry == nil) {
		return nil, errors.New("rag source catalog is required")
	}
	limit := req.Limit
	if limit <= 0 {
		limit = defaultSourceCatalogLimit
	}

	if c.registry != nil {
		sources, err := c.registry.ListSources(ctx, req)
		if err != nil {
			return nil, err
		}
		if len(sources) > 0 || c.scroller == nil {
			return sources, nil
		}
	}

	if c.scroller == nil {
		return nil, nil
	}

	filter := buildScopeFilter(req.Filter)
	bySource := map[string]*SourceSummary{}
	var offset *pb.PointId

	for {
		scrollReq := buildSourceCatalogScrollRequest(c.collection, c.pageSize, offset, filter)
		resp, err := c.scroller.Scroll(ctx, scrollReq)
		if err != nil {
			return nil, fmt.Errorf("scroll rag sources: %w", err)
		}
		for _, point := range resp.GetResult() {
			addPointToSourceCatalog(bySource, point.GetPayload())
		}
		offset = resp.GetNextPageOffset()
		if offset == nil || len(resp.GetResult()) == 0 {
			break
		}
	}

	out := make([]SourceSummary, 0, len(bySource))
	for _, source := range bySource {
		out = append(out, *source)
	}
	sort.Slice(out, func(i, j int) bool {
		leftName := strings.ToLower(strings.TrimSpace(out[i].SourceName))
		rightName := strings.ToLower(strings.TrimSpace(out[j].SourceName))
		if leftName != rightName {
			return leftName < rightName
		}
		return out[i].SourceID < out[j].SourceID
	})
	if len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}

func buildSourceCatalogScrollRequest(collection string, pageSize uint32, offset *pb.PointId, filter *pb.Filter) *pb.ScrollPoints {
	fields := []string{
		qdrantclient.FieldSourceID,
		qdrantclient.FieldSourceName,
		qdrantclient.FieldSourceURL,
		qdrantclient.FieldContentType,
		qdrantclient.FieldDocumentTitle,
		qdrantclient.FieldUserID,
		qdrantclient.FieldSessionID,
	}
	return &pb.ScrollPoints{
		CollectionName: collection,
		Filter:         filter,
		Offset:         offset,
		Limit:          &pageSize,
		WithPayload: &pb.WithPayloadSelector{
			SelectorOptions: &pb.WithPayloadSelector_Include{
				Include: &pb.PayloadIncludeSelector{Fields: fields},
			},
		},
		WithVectors: &pb.WithVectorsSelector{
			SelectorOptions: &pb.WithVectorsSelector_Enable{Enable: false},
		},
	}
}

func addPointToSourceCatalog(bySource map[string]*SourceSummary, payload map[string]*pb.Value) {
	sourceID := getPayloadString(payload, qdrantclient.FieldSourceID)
	if sourceID == "" {
		return
	}
	source := bySource[sourceID]
	if source == nil {
		source = &SourceSummary{
			SourceID:      sourceID,
			SourceName:    getPayloadString(payload, qdrantclient.FieldSourceName),
			SourceURL:     getPayloadString(payload, qdrantclient.FieldSourceURL),
			ContentType:   getPayloadString(payload, qdrantclient.FieldContentType),
			DocumentTitle: getPayloadString(payload, qdrantclient.FieldDocumentTitle),
			UserID:        getPayloadString(payload, qdrantclient.FieldUserID),
			SessionID:     getPayloadString(payload, qdrantclient.FieldSessionID),
		}
		bySource[sourceID] = source
	}
	source.ChunkCount++
	source.SourceName = firstNonEmpty(source.SourceName, getPayloadString(payload, qdrantclient.FieldSourceName))
	source.SourceURL = firstNonEmpty(source.SourceURL, getPayloadString(payload, qdrantclient.FieldSourceURL))
	source.ContentType = firstNonEmpty(source.ContentType, getPayloadString(payload, qdrantclient.FieldContentType))
	source.DocumentTitle = firstNonEmpty(source.DocumentTitle, getPayloadString(payload, qdrantclient.FieldDocumentTitle))
	source.UserID = firstNonEmpty(source.UserID, getPayloadString(payload, qdrantclient.FieldUserID))
	source.SessionID = firstNonEmpty(source.SessionID, getPayloadString(payload, qdrantclient.FieldSessionID))
}
