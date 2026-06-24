package task

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	mysqlclient "github.com/raiki02/video-extractor/internal/storage/mysql"
	"gorm.io/gorm"
)

type Repo struct {
	db *gorm.DB
}

func NewRepo(client *mysqlclient.Client) *Repo {
	return &Repo{db: client.DB}
}

// CreateTask inserts a new task and its DAG steps in a transaction.
func (r *Repo) CreateTask(ctx context.Context, t *Task, dag DAG) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		if t.Status == "" {
			t.Status = StatusPending
		}
		if t.MaxRetries == 0 {
			t.MaxRetries = 3
		}
		if err := tx.Create(t).Error; err != nil {
			return fmt.Errorf("insert task: %w", err)
		}

		for _, stepDef := range dag.Steps {
			depJSON, _ := json.Marshal(stepDef.DependsOn)
			s := Step{
				TaskID:    t.ID,
				Name:      stepDef.Name,
				Status:    StatusPending,
				DependsOn: depJSON,
			}
			if err := tx.Create(&s).Error; err != nil {
				return fmt.Errorf("insert step %s: %w", stepDef.Name, err)
			}
		}
		return nil
	})
}

// GetTask returns a task by ID.
func (r *Repo) GetTask(ctx context.Context, id string) (*Task, error) {
	var t Task
	err := r.db.WithContext(ctx).Where("id = ?", id).First(&t).Error
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, fmt.Errorf("task not found: %s", id)
		}
		return nil, fmt.Errorf("get task: %w", err)
	}
	return &t, nil
}

// GetSteps returns all steps for a task.
func (r *Repo) GetSteps(ctx context.Context, taskID string) ([]Step, error) {
	var steps []Step
	err := r.db.WithContext(ctx).
		Where("task_id = ?", taskID).
		Order("name ASC").
		Find(&steps).Error
	if err != nil {
		return nil, fmt.Errorf("get steps: %w", err)
	}
	return steps, nil
}

// UpdateTaskStatus updates the status and optionally output/error.
func (r *Repo) UpdateTaskStatus(ctx context.Context, id string, status Status, output json.RawMessage, errMsg string) error {
	updates := map[string]any{
		"status":     status,
		"updated_at": time.Now(),
	}
	if output != nil {
		updates["output"] = output
	}
	if errMsg != "" {
		updates["error_msg"] = errMsg
	}
	return r.db.WithContext(ctx).Model(&Task{}).Where("id = ?", id).Updates(updates).Error
}

// UpdateStepStatus updates a single step's status.
func (r *Repo) UpdateStepStatus(ctx context.Context, id string, status Status, output json.RawMessage, errMsg string) error {
	updates := map[string]any{"status": status}
	now := time.Now()

	switch status {
	case StatusRunning:
		updates["started_at"] = now
	case StatusDone, StatusFailed:
		updates["finished_at"] = now
	}
	if output != nil {
		updates["output"] = output
	}
	if errMsg != "" {
		updates["error_msg"] = errMsg
	}
	return r.db.WithContext(ctx).Model(&Step{}).Where("id = ?", id).Updates(updates).Error
}

// IncrementTaskRetry increments retry count for a task.
func (r *Repo) IncrementTaskRetry(ctx context.Context, id string) error {
	return r.db.WithContext(ctx).Model(&Task{}).Where("id = ?", id).
		UpdateColumn("retry_count", gorm.Expr("retry_count + 1")).Error
}

// CreateTranscript inserts a transcript record.
func (r *Repo) CreateTranscript(ctx context.Context, t *Transcript) error {
	return r.db.WithContext(ctx).Create(t).Error
}

// LogToolExecution inserts a tool execution log.
func (r *Repo) LogToolExecution(ctx context.Context, entry *ToolLog) error {
	return r.db.WithContext(ctx).Create(entry).Error
}
