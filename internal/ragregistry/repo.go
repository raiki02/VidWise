package ragregistry

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/raiki02/vidwise/internal/rag"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

// Repo is the MySQL adapter for the RAG source registry.
type Repo struct {
	db *gorm.DB
}

func NewRepo(db *gorm.DB) *Repo {
	return &Repo{db: db}
}

func (r *Repo) AutoMigrate() error {
	return r.db.AutoMigrate(&SourceRecord{})
}

func (r *Repo) RecordIndexed(ctx context.Context, sources []rag.SourceSummary) error {
	if r == nil || r.db == nil || len(sources) == 0 {
		return nil
	}

	records := make([]SourceRecord, 0, len(sources))
	for _, source := range sources {
		record, ok := sourceRecordFromSummary(source)
		if ok {
			records = append(records, record)
		}
	}
	if len(records) == 0 {
		return nil
	}

	if err := r.db.WithContext(ctx).Clauses(clause.OnConflict{
		Columns: []clause.Column{{Name: "source_id"}},
		DoUpdates: clause.AssignmentColumns([]string{
			"user_id",
			"session_id",
			"source_name",
			"source_url",
			"content_type",
			"document_title",
			"chunk_count",
			"status",
			"deleted_at",
			"updated_at",
		}),
	}).Create(&records).Error; err != nil {
		return fmt.Errorf("record rag sources: %w", err)
	}
	return nil
}

func (r *Repo) MarkDeleted(ctx context.Context, req rag.DeleteRequest) error {
	if r == nil || r.db == nil {
		return nil
	}
	sourceIDs := normalizeSourceIDs(req.SourceIDs)
	if len(sourceIDs) == 0 {
		return nil
	}

	now := time.Now()
	query := r.db.WithContext(ctx).Model(&SourceRecord{}).
		Where("source_id IN ?", sourceIDs)
	if req.Filter != nil {
		if req.Filter.UserID != "" {
			query = query.Where("user_id = ?", strings.TrimSpace(req.Filter.UserID))
		}
		if req.Filter.SessionID != "" {
			query = query.Where("session_id = ?", strings.TrimSpace(req.Filter.SessionID))
		}
	}
	if err := query.Updates(map[string]any{
		"status":     StatusDeleted,
		"deleted_at": &now,
	}).Error; err != nil {
		return fmt.Errorf("mark rag sources deleted: %w", err)
	}
	return nil
}

func (r *Repo) ListSources(ctx context.Context, req rag.SourceListRequest) ([]rag.SourceSummary, error) {
	if r == nil || r.db == nil {
		return nil, nil
	}
	limit := req.Limit
	if limit <= 0 {
		limit = 50
	}

	query := r.db.WithContext(ctx).Model(&SourceRecord{}).
		Where("status = ?", StatusActive)
	if req.Filter != nil {
		if req.Filter.UserID != "" {
			query = query.Where("user_id = ?", strings.TrimSpace(req.Filter.UserID))
		}
		if req.Filter.SessionID != "" {
			query = query.Where("session_id = ?", strings.TrimSpace(req.Filter.SessionID))
		}
	}

	var records []SourceRecord
	if err := query.Order("updated_at DESC").Limit(limit).Find(&records).Error; err != nil {
		return nil, fmt.Errorf("list rag sources: %w", err)
	}

	out := make([]rag.SourceSummary, 0, len(records))
	for _, record := range records {
		out = append(out, sourceSummaryFromRecord(record))
	}
	return out, nil
}

func sourceRecordFromSummary(source rag.SourceSummary) (SourceRecord, bool) {
	sourceID := strings.TrimSpace(source.SourceID)
	if sourceID == "" {
		return SourceRecord{}, false
	}
	return SourceRecord{
		SourceID:      sourceID,
		UserID:        strings.TrimSpace(source.UserID),
		SessionID:     strings.TrimSpace(source.SessionID),
		SourceName:    strings.TrimSpace(source.SourceName),
		SourceURL:     strings.TrimSpace(source.SourceURL),
		ContentType:   strings.TrimSpace(source.ContentType),
		DocumentTitle: strings.TrimSpace(source.DocumentTitle),
		ChunkCount:    source.ChunkCount,
		Status:        StatusActive,
		DeletedAt:     nil,
	}, true
}

func sourceSummaryFromRecord(record SourceRecord) rag.SourceSummary {
	return rag.SourceSummary{
		SourceID:      record.SourceID,
		SourceName:    record.SourceName,
		SourceURL:     record.SourceURL,
		ContentType:   record.ContentType,
		DocumentTitle: record.DocumentTitle,
		UserID:        record.UserID,
		SessionID:     record.SessionID,
		ChunkCount:    record.ChunkCount,
	}
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
