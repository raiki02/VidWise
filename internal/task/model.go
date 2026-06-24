package task

import (
	"encoding/json"
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

type Status string

const (
	StatusPending Status = "pending"
	StatusRunning Status = "running"
	StatusDone    Status = "done"
	StatusFailed  Status = "failed"
)

// Task represents an asynchronous video processing or chat task.
type Task struct {
	ID         string          `gorm:"type:varchar(36);primaryKey" json:"id"`
	UserID     string          `gorm:"type:varchar(36);not null;index:idx_tasks_user" json:"user_id"`
	SessionID  string          `gorm:"type:varchar(36);not null;index:idx_tasks_session" json:"session_id"`
	Type       string          `gorm:"type:varchar(64);not null" json:"type"`
	Status     Status          `gorm:"type:varchar(16);not null;default:pending;index:idx_tasks_status" json:"status"`
	Input      json.RawMessage `gorm:"type:json" json:"input,omitempty"`
	Output     json.RawMessage `gorm:"type:json" json:"output,omitempty"`
	RetryCount int             `gorm:"not null;default:0" json:"retry_count"`
	MaxRetries int             `gorm:"not null;default:3" json:"max_retries"`
	TraceID    *string         `gorm:"type:varchar(64)" json:"trace_id,omitempty"`
	ErrorMsg   *string         `gorm:"type:text" json:"error_msg,omitempty"`
	CreatedAt  time.Time       `gorm:"autoCreateTime" json:"created_at"`
	UpdatedAt  time.Time       `gorm:"autoUpdateTime" json:"updated_at"`
}

// Step represents a single step in a task's DAG.
type Step struct {
	ID         string          `gorm:"type:varchar(36);primaryKey" json:"id"`
	TaskID     string          `gorm:"type:varchar(36);not null;index:idx_steps_task" json:"task_id"`
	Name       string          `gorm:"type:varchar(128);not null" json:"name"`
	Status     Status          `gorm:"type:varchar(16);not null;default:pending" json:"status"`
	DependsOn  json.RawMessage `gorm:"type:json" json:"depends_on,omitempty"`
	Input      json.RawMessage `gorm:"type:json" json:"input,omitempty"`
	Output     json.RawMessage `gorm:"type:json" json:"output,omitempty"`
	RetryCount int             `gorm:"not null;default:0" json:"retry_count"`
	ErrorMsg   *string         `gorm:"type:text" json:"error_msg,omitempty"`
	StartedAt  *time.Time      `json:"started_at,omitempty"`
	FinishedAt *time.Time      `json:"finished_at,omitempty"`
}

// Transcript stores the ASR output for a task.
type Transcript struct {
	ID            string    `gorm:"type:varchar(36);primaryKey" json:"id"`
	TaskID        string    `gorm:"type:varchar(36);not null;index" json:"task_id"`
	SessionID     string    `gorm:"type:varchar(36);not null;index:idx_transcripts_session" json:"session_id"`
	UserID        string    `gorm:"type:varchar(36);not null" json:"user_id"`
	RawText       *string   `gorm:"type:mediumtext" json:"raw_text,omitempty"`
	FormattedText *string   `gorm:"type:mediumtext" json:"formatted_text,omitempty"`
	Language      *string   `gorm:"type:varchar(16)" json:"language,omitempty"`
	DurationSec   *float64  `json:"duration_sec,omitempty"`
	CreatedAt     time.Time `gorm:"autoCreateTime" json:"created_at"`
}

// ToolLog records a single tool invocation for audit and debugging.
type ToolLog struct {
	ID         string          `gorm:"type:varchar(36);primaryKey" json:"id"`
	TaskID     string          `gorm:"type:varchar(36);not null;index:idx_tool_log_task" json:"task_id"`
	StepID     *string         `gorm:"type:varchar(36)" json:"step_id,omitempty"`
	ToolName   string          `gorm:"type:varchar(128);not null" json:"tool_name"`
	Input      json.RawMessage `gorm:"type:json" json:"input,omitempty"`
	Output     json.RawMessage `gorm:"type:json" json:"output,omitempty"`
	DurationMs int64           `gorm:"" json:"duration_ms"`
	Status     string          `gorm:"type:varchar(16);not null" json:"status"`
	ErrorMsg   *string         `gorm:"type:text" json:"error_msg,omitempty"`
	TraceID    *string         `gorm:"type:varchar(64);index:idx_tool_log_trace" json:"trace_id,omitempty"`
	CreatedAt  time.Time       `gorm:"autoCreateTime" json:"created_at"`
}

// DAG represents the task execution DAG.
type DAG struct {
	Steps []StepDef `json:"steps"`
}

type StepDef struct {
	Name      string   `json:"name"`
	DependsOn []string `json:"depends_on,omitempty"`
}

// TableName overrides.
func (Task) TableName() string       { return "tasks" }
func (Step) TableName() string       { return "task_steps" }
func (Transcript) TableName() string { return "transcripts" }
func (ToolLog) TableName() string    { return "tool_execution_log" }

// newUUID is the shared UUID generator for this package.
func newUUID() string { return uuid.New().String() }

// BeforeCreate hooks auto-generate UUIDs.
func (t *Task) BeforeCreate(tx *gorm.DB) error {
	if t.ID == "" {
		t.ID = newUUID()
	}
	return nil
}

func (s *Step) BeforeCreate(tx *gorm.DB) error {
	if s.ID == "" {
		s.ID = newUUID()
	}
	return nil
}

func (tr *Transcript) BeforeCreate(tx *gorm.DB) error {
	if tr.ID == "" {
		tr.ID = newUUID()
	}
	return nil
}

func (tl *ToolLog) BeforeCreate(tx *gorm.DB) error {
	if tl.ID == "" {
		tl.ID = newUUID()
	}
	return nil
}
