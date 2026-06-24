package task

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
	mysqlclient "github.com/raiki02/video-extractor/internal/storage/mysql"
)

type Repo struct {
	db *sql.DB
}

func NewRepo(client *mysqlclient.Client) *Repo {
	return &Repo{db: client.DB}
}

// CreateTask inserts a new task and its DAG steps in a transaction.
func (r *Repo) CreateTask(ctx context.Context, t *Task, dag DAG) error {
	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	if t.ID == "" {
		t.ID = uuid.New().String()
	}
	if t.Status == "" {
		t.Status = StatusPending
	}
	t.MaxRetries = 3

	_, err = tx.ExecContext(ctx,
		`INSERT INTO tasks (id, user_id, session_id, type, status, input, max_retries, trace_id)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		t.ID, t.UserID, t.SessionID, t.Type, t.Status, t.Input, t.MaxRetries, t.TraceID,
	)
	if err != nil {
		return fmt.Errorf("insert task: %w", err)
	}

	for _, stepDef := range dag.Steps {
		depJSON, _ := json.Marshal(stepDef.DependsOn)
		s := Step{
			ID:        uuid.New().String(),
			TaskID:    t.ID,
			Name:      stepDef.Name,
			Status:    StatusPending,
			DependsOn: depJSON,
		}
		_, err = tx.ExecContext(ctx,
			`INSERT INTO task_steps (id, task_id, name, status, depends_on)
			 VALUES (?, ?, ?, ?, ?)`,
			s.ID, s.TaskID, s.Name, s.Status, s.DependsOn,
		)
		if err != nil {
			return fmt.Errorf("insert step %s: %w", stepDef.Name, err)
		}
	}
	return tx.Commit()
}

// GetTask returns a task by ID.
func (r *Repo) GetTask(ctx context.Context, id string) (*Task, error) {
	var t Task
	err := r.db.QueryRowContext(ctx,
		`SELECT id, user_id, session_id, type, status, input, output,
		        retry_count, max_retries, trace_id, error_msg, created_at, updated_at
		 FROM tasks WHERE id = ?`, id,
	).Scan(&t.ID, &t.UserID, &t.SessionID, &t.Type, &t.Status,
		&t.Input, &t.Output, &t.RetryCount, &t.MaxRetries,
		&t.TraceID, &t.ErrorMsg, &t.CreatedAt, &t.UpdatedAt)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("task not found: %s", id)
		}
		return nil, fmt.Errorf("get task: %w", err)
	}
	return &t, nil
}

// GetSteps returns all steps for a task.
func (r *Repo) GetSteps(ctx context.Context, taskID string) ([]*Step, error) {
	rows, err := r.db.QueryContext(ctx,
		`SELECT id, task_id, name, status, depends_on, input, output,
		        retry_count, error_msg, started_at, finished_at
		 FROM task_steps WHERE task_id = ? ORDER BY name`, taskID,
	)
	if err != nil {
		return nil, fmt.Errorf("get steps: %w", err)
	}
	defer rows.Close()

	var steps []*Step
	for rows.Next() {
		var s Step
		if err := rows.Scan(&s.ID, &s.TaskID, &s.Name, &s.Status,
			&s.DependsOn, &s.Input, &s.Output, &s.RetryCount,
			&s.ErrorMsg, &s.StartedAt, &s.FinishedAt,
		); err != nil {
			return nil, fmt.Errorf("scan step: %w", err)
		}
		steps = append(steps, &s)
	}
	return steps, rows.Err()
}

// UpdateTaskStatus updates the status and optionally output/error.
func (r *Repo) UpdateTaskStatus(ctx context.Context, id string, status Status, output json.RawMessage, errMsg string) error {
	q := "UPDATE tasks SET status = ?, updated_at = ?"
	args := []any{status, time.Now()}
	if output != nil {
		q += ", output = ?"
		args = append(args, output)
	}
	if errMsg != "" {
		q += ", error_msg = ?"
		args = append(args, errMsg)
	}
	q += " WHERE id = ?"
	args = append(args, id)

	_, err := r.db.ExecContext(ctx, q, args...)
	if err != nil {
		return fmt.Errorf("update task status: %w", err)
	}
	return nil
}

// UpdateStepStatus updates a single step's status.
func (r *Repo) UpdateStepStatus(ctx context.Context, id string, status Status, output json.RawMessage, errMsg string) error {
	q := "UPDATE task_steps SET status = ?"
	args := []any{status}
	if status == StatusRunning {
		q += ", started_at = ?"
		args = append(args, time.Now())
	}
	if status == StatusDone || status == StatusFailed {
		q += ", finished_at = ?"
		args = append(args, time.Now())
	}
	if output != nil {
		q += ", output = ?"
		args = append(args, output)
	}
	if errMsg != "" {
		q += ", error_msg = ?"
		args = append(args, errMsg)
	}
	q += " WHERE id = ?"
	args = append(args, id)

	_, err := r.db.ExecContext(ctx, q, args...)
	if err != nil {
		return fmt.Errorf("update step %s: %w", id, err)
	}
	return nil
}

// IncrementRetry increments retry count for a task.
func (r *Repo) IncrementTaskRetry(ctx context.Context, id string) error {
	_, err := r.db.ExecContext(ctx,
		"UPDATE tasks SET retry_count = retry_count + 1 WHERE id = ?", id)
	return err
}

// CreateTranscript inserts a transcript record.
func (r *Repo) CreateTranscript(ctx context.Context, id, taskID, sessionID, userID string, rawText, formattedText, language string, durationSec float64) error {
	_, err := r.db.ExecContext(ctx,
		`INSERT INTO transcripts (id, task_id, session_id, user_id, raw_text, formatted_text, language, duration_sec)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		id, taskID, sessionID, userID, rawText, formattedText, language, durationSec)
	return err
}

// LogToolExecution inserts a tool execution log.
func (r *Repo) LogToolExecution(ctx context.Context, entry ToolLog) error {
	if entry.ID == "" {
		entry.ID = uuid.New().String()
	}
	_, err := r.db.ExecContext(ctx,
		`INSERT INTO tool_execution_log (id, task_id, step_id, tool_name, input, output, duration_ms, status, error_msg, trace_id)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		entry.ID, entry.TaskID, entry.StepID, entry.ToolName,
		entry.Input, entry.Output, entry.DurationMs,
		entry.Status, entry.ErrorMsg, entry.TraceID,
	)
	return err
}

type ToolLog struct {
	ID         string          `json:"id"`
	TaskID     string          `json:"task_id"`
	StepID     sql.NullString  `json:"step_id,omitempty"`
	ToolName   string          `json:"tool_name"`
	Input      json.RawMessage `json:"input,omitempty"`
	Output     json.RawMessage `json:"output,omitempty"`
	DurationMs int64           `json:"duration_ms"`
	Status     string          `json:"status"`
	ErrorMsg   sql.NullString  `json:"error_msg,omitempty"`
	TraceID    sql.NullString  `json:"trace_id,omitempty"`
}
