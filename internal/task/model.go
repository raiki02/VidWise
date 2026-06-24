package task

import (
	"database/sql"
	"encoding/json"
	"time"
)

type Status string

const (
	StatusPending Status = "pending"
	StatusRunning Status = "running"
	StatusDone    Status = "done"
	StatusFailed  Status = "failed"
)

type Task struct {
	ID         string          `json:"id"`
	UserID     string          `json:"user_id"`
	SessionID  string          `json:"session_id"`
	Type       string          `json:"type"`
	Status     Status          `json:"status"`
	Input      json.RawMessage `json:"input,omitempty"`
	Output     json.RawMessage `json:"output,omitempty"`
	RetryCount int             `json:"retry_count"`
	MaxRetries int             `json:"max_retries"`
	TraceID    sql.NullString  `json:"trace_id,omitempty"`
	ErrorMsg   sql.NullString  `json:"error_msg,omitempty"`
	CreatedAt  time.Time       `json:"created_at"`
	UpdatedAt  time.Time       `json:"updated_at"`
}

type Step struct {
	ID         string          `json:"id"`
	TaskID     string          `json:"task_id"`
	Name       string          `json:"name"`
	Status     Status          `json:"status"`
	DependsOn  json.RawMessage `json:"depends_on,omitempty"`
	Input      json.RawMessage `json:"input,omitempty"`
	Output     json.RawMessage `json:"output,omitempty"`
	RetryCount int             `json:"retry_count"`
	ErrorMsg   sql.NullString  `json:"error_msg,omitempty"`
	StartedAt  sql.NullTime    `json:"started_at,omitempty"`
	FinishedAt sql.NullTime    `json:"finished_at,omitempty"`
}

// DAG represents the task execution DAG.
type DAG struct {
	Steps []StepDef `json:"steps"`
}

type StepDef struct {
	Name      string   `json:"name"`
	DependsOn []string `json:"depends_on,omitempty"`
}
