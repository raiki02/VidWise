package qdrant

import (
	"context"
	"fmt"
	"log/slog"

	pb "github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	FieldText              = "text"
	FieldTaskID            = "task_id"
	FieldUserID            = "user_id"
	FieldSessionID         = "session_id"
	FieldChunkIdx          = "chunk_index"
	FieldSourceID          = "source_id"
	FieldDocumentID        = "document_id"
	FieldContentHash       = "content_hash"
	FieldChunkID           = "chunk_id"
	FieldContentType       = "content_type"
	FieldSourceName        = "source_name"
	FieldSourceURL         = "source_url"
	FieldDocumentTitle     = "document_title"
	FieldHeadingPath       = "heading_path"
	FieldHeadingLevel      = "heading_level"
	FieldSectionIndex      = "section_index"
	FieldSectionChunkIndex = "section_chunk_index"
	FieldChunkSource       = "chunk_source"
	FieldHeader1           = "header_1"
	FieldHeader2           = "header_2"
	FieldHeader3           = "header_3"
	FieldHeader4           = "header_4"
	FieldHeader5           = "header_5"
	FieldHeader6           = "header_6"
)

func EnsureCollection(ctx context.Context, c *Client, name string, vectorDim uint64) error {
	// Check if collection exists
	list, err := c.Collections.List(ctx, &pb.ListCollectionsRequest{})
	if err != nil {
		return fmt.Errorf("list collections: %w", err)
	}
	for _, col := range list.Collections {
		if col.Name == name {
			slog.Info("qdrant.collection.exists", "name", name)
			return EnsurePayloadIndexes(ctx, c, name)
		}
	}

	slog.Info("qdrant.collection.create", "name", name, "dim", vectorDim)
	_, err = c.Collections.Create(ctx, &pb.CreateCollection{
		CollectionName: name,
		VectorsConfig: &pb.VectorsConfig{
			Config: &pb.VectorsConfig_Params{
				Params: &pb.VectorParams{
					Size:     vectorDim,
					Distance: pb.Distance_Cosine,
				},
			},
		},
	})
	if err != nil {
		return fmt.Errorf("create collection %s: %w", name, err)
	}

	return EnsurePayloadIndexes(ctx, c, name)
}

type payloadIndexSpec struct {
	field     string
	fieldType pb.FieldType
}

func payloadIndexes() []payloadIndexSpec {
	return []payloadIndexSpec{
		{FieldTaskID, pb.FieldType_FieldTypeKeyword},
		{FieldUserID, pb.FieldType_FieldTypeKeyword},
		{FieldSessionID, pb.FieldType_FieldTypeKeyword},
		{FieldChunkIdx, pb.FieldType_FieldTypeInteger},
		{FieldSourceID, pb.FieldType_FieldTypeKeyword},
		{FieldDocumentID, pb.FieldType_FieldTypeKeyword},
		{FieldContentHash, pb.FieldType_FieldTypeKeyword},
		{FieldChunkID, pb.FieldType_FieldTypeKeyword},
		{FieldContentType, pb.FieldType_FieldTypeKeyword},
		{FieldSourceName, pb.FieldType_FieldTypeKeyword},
		{FieldSourceURL, pb.FieldType_FieldTypeKeyword},
		{FieldDocumentTitle, pb.FieldType_FieldTypeKeyword},
		{FieldHeadingPath, pb.FieldType_FieldTypeKeyword},
		{FieldHeadingLevel, pb.FieldType_FieldTypeInteger},
		{FieldSectionIndex, pb.FieldType_FieldTypeInteger},
		{FieldSectionChunkIndex, pb.FieldType_FieldTypeInteger},
		{FieldChunkSource, pb.FieldType_FieldTypeKeyword},
		{FieldHeader1, pb.FieldType_FieldTypeKeyword},
		{FieldHeader2, pb.FieldType_FieldTypeKeyword},
		{FieldHeader3, pb.FieldType_FieldTypeKeyword},
		{FieldHeader4, pb.FieldType_FieldTypeKeyword},
		{FieldHeader5, pb.FieldType_FieldTypeKeyword},
		{FieldHeader6, pb.FieldType_FieldTypeKeyword},
	}
}

func EnsurePayloadIndexes(ctx context.Context, c *Client, name string) error {
	for _, idx := range payloadIndexes() {
		_, err := c.Points.CreateFieldIndex(ctx, &pb.CreateFieldIndexCollection{
			CollectionName: name,
			FieldName:      idx.field,
			FieldType:      &idx.fieldType,
		})
		if err != nil {
			if status.Code(err) == codes.AlreadyExists {
				continue
			}
			slog.Warn("qdrant.index.create_failed", "field", idx.field, "err", err)
		}
	}
	return nil
}
