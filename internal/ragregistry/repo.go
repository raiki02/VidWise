package ragregistry

import (
	"context"
	"encoding/json"
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
			"document_ids",
			"task_ids",
			"heading_paths",
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
		DocumentIDs:   encodeStringList(source.DocumentIDs),
		TaskIDs:       encodeStringList(source.TaskIDs),
		HeadingPaths:  encodeStringList(source.HeadingPaths),
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
		DocumentIDs:   decodeStringList(record.DocumentIDs),
		TaskIDs:       decodeStringList(record.TaskIDs),
		HeadingPaths:  decodeStringList(record.HeadingPaths),
		UserID:        record.UserID,
		SessionID:     record.SessionID,
		ChunkCount:    record.ChunkCount,
	}
}

func encodeStringList(values []string) string {
	normalized := normalizeStringList(values)
	if len(normalized) == 0 {
		return ""
	}
	data, err := json.Marshal(normalized)
	if err != nil {
		return ""
	}
	return string(data)
}

func decodeStringList(raw string) []string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	var values []string
	if err := json.Unmarshal([]byte(raw), &values); err != nil {
		return nil
	}
	return normalizeStringList(values)
}

func normalizeStringList(values []string) []string {
	out := make([]string, 0, len(values))
	seen := map[string]bool{}
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" || seen[value] {
			continue
		}
		out = append(out, value)
		seen[value] = true
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
