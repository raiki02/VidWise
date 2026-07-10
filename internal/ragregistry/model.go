package ragregistry

import (
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

const (
	StatusActive  = "active"
	StatusDeleted = "deleted"
)

// SourceRecord is the durable operational record for one indexed RAG source.
type SourceRecord struct {
	ID            string     `gorm:"type:varchar(36);primaryKey" json:"id"`
	SourceID      string     `gorm:"type:varchar(128);not null;uniqueIndex" json:"source_id"`
	UserID        string     `gorm:"type:varchar(64);index:idx_rag_sources_user_status" json:"user_id,omitempty"`
	SessionID     string     `gorm:"type:varchar(64);index:idx_rag_sources_session_status" json:"session_id,omitempty"`
	SourceName    string     `gorm:"type:varchar(512)" json:"source_name,omitempty"`
	SourceURL     string     `gorm:"type:text" json:"source_url,omitempty"`
	ContentType   string     `gorm:"type:varchar(64);index" json:"content_type,omitempty"`
	DocumentTitle string     `gorm:"type:varchar(512)" json:"document_title,omitempty"`
	DocumentIDs   string     `gorm:"type:text" json:"document_ids,omitempty"`
	TaskIDs       string     `gorm:"type:text" json:"task_ids,omitempty"`
	HeadingPaths  string     `gorm:"type:text" json:"heading_paths,omitempty"`
	ChunkCount    int        `gorm:"not null;default:0" json:"chunk_count"`
	Status        string     `gorm:"type:varchar(16);not null;default:active;index:idx_rag_sources_user_status;index:idx_rag_sources_session_status" json:"status"`
	DeletedAt     *time.Time `json:"deleted_at,omitempty"`
	CreatedAt     time.Time  `json:"created_at"`
	UpdatedAt     time.Time  `json:"updated_at"`
}

func (SourceRecord) TableName() string {
	return "rag_sources"
}

func (r *SourceRecord) BeforeCreate(tx *gorm.DB) error {
	if r.ID == "" {
		r.ID = uuid.New().String()
	}
	if r.Status == "" {
		r.Status = StatusActive
	}
	return nil
}
