package memory

import (
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

// FactSource records where a memory fact was derived from.
type FactSource string

const (
	FactSourceExplicit  FactSource = "explicit"  // User explicitly stated this.
	FactSourceInferred  FactSource = "inferred"  // LLM inferred this from conversation.
	FactSourceCorrected FactSource = "corrected" // LLM corrected a previous fact.
)

// ConfidenceLevel represents how confident the system is in a fact.
type ConfidenceLevel string

const (
	ConfidenceHigh   ConfidenceLevel = "high"
	ConfidenceMedium ConfidenceLevel = "medium"
	ConfidenceLow    ConfidenceLevel = "low"
)

// MemoryFact stores a single extracted fact about a user for cross-session memory.
type MemoryFact struct {
	ID          string `gorm:"type:varchar(36);primaryKey" json:"id"`
	UserID      string `gorm:"type:varchar(36);not null;index:idx_mem_user" json:"user_id"`
	Category    string `gorm:"type:varchar(64);not null;index:idx_mem_cat" json:"category"`
	Key         string `gorm:"type:varchar(255);not null" json:"key"`
	Value       string `gorm:"type:text;not null" json:"value"`
	Evidence    string `gorm:"type:text" json:"evidence"`
	Source      FactSource `gorm:"type:varchar(16);not null;default:explicit" json:"source"`
	Confidence  ConfidenceLevel `gorm:"type:varchar(16);not null;default:high" json:"confidence"`

	// Conflict resolution fields
	SupersededBy *string `gorm:"type:varchar(36)" json:"superseded_by,omitempty"`
	SupersededAt *time.Time        `json:"superseded_at,omitempty"`
	RevisionNote *string           `gorm:"type:text" json:"revision_note,omitempty"`

	// We track contexts where the fact was mentioned to help with conflict resolution
	MentionCount int       `json:"mention_count"`
	LastMentionedAt time.Time `json:"last_mentioned_at"`
	ExtractedFrom   string    `gorm:"type:varchar(36)" json:"extracted_from"` // session_id

	CreatedAt time.Time `gorm:"autoCreateTime" json:"created_at"`
	UpdatedAt time.Time `gorm:"autoUpdateTime" json:"updated_at"`
}

func (MemoryFact) TableName() string { return "memory_facts" }

func (m *MemoryFact) BeforeCreate(tx *gorm.DB) error {
	if m.ID == "" {
		m.ID = uuid.New().String()
	}
	if m.LastMentionedAt.IsZero() {
		m.LastMentionedAt = time.Now()
	}
	return nil
}

// MemoryExtractionInput is the input for the LLM to extract user facts from a turn.
type MemoryExtractionInput struct {
	ExistingFacts  []MemoryFactSummary `json:"existing_facts"`
	UserMessage    string              `json:"user_message"`
	AssistantReply string              `json:"assistant_reply"`
	RecentHistory  string              `json:"recent_history"`
}

// MemoryFactSummary is a lightweight view of existing facts for the LLM prompt.
type MemoryFactSummary struct {
	ID         string `json:"id"`
	Category   string `json:"category"`
	Key        string `json:"key"`
	Value      string `json:"value"`
	Confidence string `json:"confidence"`
}

// ExtractedFact is the LLM output for a single extracted/updated fact.
type ExtractedFact struct {
	Category   string           `json:"category"`
	Key        string           `json:"key"`
	Value      string           `json:"value"`
	Evidence   string           `json:"evidence"`
	Source     FactSource       `json:"source"`
	Confidence ConfidenceLevel  `json:"confidence"`
	Action     string           `json:"action"` // "create", "update", "supersede", "confirm"
	TargetID   string           `json:"target_id,omitempty"` // For update/supersede: which existing fact
	Reason     string           `json:"reason,omitempty"`    // For supersede: why the new fact is more reliable
}

// MemoryExtractionResult is the structured output from the LLM memory extraction.
type MemoryExtractionResult struct {
	Facts  []ExtractedFact `json:"facts"`
	Notes  string           `json:"notes,omitempty"`
}
