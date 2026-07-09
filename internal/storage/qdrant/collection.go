package qdrant

import (
	"context"
	"fmt"
	"log/slog"

	pb "github.com/qdrant/go-client/qdrant"
)

const (
	FieldText              = "text"
	FieldTaskID            = "task_id"
	FieldUserID            = "user_id"
	FieldSessionID         = "session_id"
	FieldChunkIdx          = "chunk_index"
	FieldContentType       = "content_type"
	FieldSourceName        = "source_name"
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
			return nil
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

	// Create payload indexes for metadata filtering
	indexes := []struct {
		field     string
		fieldType pb.FieldType
	}{
		{FieldTaskID, pb.FieldType_FieldTypeKeyword},
		{FieldUserID, pb.FieldType_FieldTypeKeyword},
		{FieldSessionID, pb.FieldType_FieldTypeKeyword},
		{FieldChunkIdx, pb.FieldType_FieldTypeInteger},
		{FieldContentType, pb.FieldType_FieldTypeKeyword},
		{FieldSourceName, pb.FieldType_FieldTypeKeyword},
		{FieldDocumentTitle, pb.FieldType_FieldTypeKeyword},
		{FieldHeadingPath, pb.FieldType_FieldTypeKeyword},
		{FieldHeadingLevel, pb.FieldType_FieldTypeInteger},
		{FieldSectionIndex, pb.FieldType_FieldTypeInteger},
		{FieldSectionChunkIndex, pb.FieldType_FieldTypeInteger},
		{FieldChunkSource, pb.FieldType_FieldTypeKeyword},
	}

	for _, idx := range indexes {
		_, err := c.Points.CreateFieldIndex(ctx, &pb.CreateFieldIndexCollection{
			CollectionName: name,
			FieldName:      idx.field,
			FieldType:      &idx.fieldType,
		})
		if err != nil {
			slog.Warn("qdrant.index.create_failed", "field", idx.field, "err", err)
		}
	}
	return nil
}
