package task

import (
	"context"
	"encoding/json"
	"log/slog"

	"github.com/google/uuid"
)

// Manager orchestrates task lifecycle: create, query, cancel.
type Manager struct {
	repo *Repo
}

func NewManager(repo *Repo) *Manager {
	return &Manager{repo: repo}
}

// CreateTask creates a new task with the given DAG and returns the task ID.
func (m *Manager) CreateTask(ctx context.Context, userID, sessionID, taskType string, input any, dagDef []StepDef, traceID string) (string, error) {
	inputJSON, _ := json.Marshal(input)

	t := &Task{
		ID:        uuid.New().String(),
		UserID:    userID,
		SessionID: sessionID,
		Type:      taskType,
		Input:     inputJSON,
	}

	dag := DAG{Steps: dagDef}
	if err := m.repo.CreateTask(ctx, t, dag); err != nil {
		return "", err
	}
	slog.Info("task.created", "task_id", t.ID, "type", taskType, "trace_id", traceID)
	return t.ID, nil
}

// GetTask returns a task by ID.
func (m *Manager) GetTask(ctx context.Context, id string) (*Task, error) {
	return m.repo.GetTask(ctx, id)
}

// GetTaskWithSteps returns a task and its steps.
func (m *Manager) GetTaskWithSteps(ctx context.Context, id string) (*Task, []*Step, error) {
	t, err := m.repo.GetTask(ctx, id)
	if err != nil {
		return nil, nil, err
	}
	steps, err := m.repo.GetSteps(ctx, id)
	if err != nil {
		return t, nil, err
	}
	return t, steps, nil
}

// BeginTask marks a task as running.
func (m *Manager) BeginTask(ctx context.Context, taskID string) error {
	return m.repo.UpdateTaskStatus(ctx, taskID, StatusRunning, nil, "")
}

// CompleteTask marks a task as done.
func (m *Manager) CompleteTask(ctx context.Context, taskID string, output any) error {
	outputJSON, _ := json.Marshal(output)
	return m.repo.UpdateTaskStatus(ctx, taskID, StatusDone, outputJSON, "")
}

// FailTask marks a task as failed.
func (m *Manager) FailTask(ctx context.Context, taskID, errMsg string) error {
	return m.repo.UpdateTaskStatus(ctx, taskID, StatusFailed, nil, errMsg)
}
