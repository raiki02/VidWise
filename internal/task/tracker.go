package task

import (
	"sync"
	"time"
)

// TrackedTask is the request-facing view of an asynchronous task.
type TrackedTask struct {
	ID         string         `json:"task_id"`
	Type       string         `json:"type"`
	Status     Status         `json:"status"`
	UserID     string         `json:"user_id,omitempty"`
	SessionID  string         `json:"session_id,omitempty"`
	TraceID    string         `json:"trace_id,omitempty"`
	Error      string         `json:"error,omitempty"`
	Output     map[string]any `json:"output,omitempty"`
	CreatedAt  time.Time      `json:"created_at"`
	UpdatedAt  time.Time      `json:"updated_at"`
	StartedAt  *time.Time     `json:"started_at,omitempty"`
	FinishedAt *time.Time     `json:"finished_at,omitempty"`
}

type TrackCreateRequest struct {
	ID        string
	Type      string
	UserID    string
	SessionID string
	TraceID   string
}

// Tracker records in-process task state for async work launched by the gateway.
type Tracker struct {
	mu    sync.RWMutex
	tasks map[string]TrackedTask
	now   func() time.Time
}

func NewTracker() *Tracker {
	return &Tracker{
		tasks: make(map[string]TrackedTask),
		now:   time.Now,
	}
}

func (t *Tracker) Create(req TrackCreateRequest) TrackedTask {
	if t == nil {
		return TrackedTask{}
	}
	now := t.currentTime()
	id := req.ID
	if id == "" {
		id = newUUID()
	}
	task := TrackedTask{
		ID:        id,
		Type:      req.Type,
		Status:    StatusPending,
		UserID:    req.UserID,
		SessionID: req.SessionID,
		TraceID:   req.TraceID,
		CreatedAt: now,
		UpdatedAt: now,
	}

	t.mu.Lock()
	defer t.mu.Unlock()
	if t.tasks == nil {
		t.tasks = make(map[string]TrackedTask)
	}
	t.tasks[id] = task
	return copyTrackedTask(task)
}

func (t *Tracker) Start(id string) (TrackedTask, bool) {
	return t.update(id, func(task TrackedTask, now time.Time) TrackedTask {
		task.Status = StatusRunning
		task.StartedAt = &now
		task.UpdatedAt = now
		return task
	})
}

func (t *Tracker) Complete(id string, output map[string]any) (TrackedTask, bool) {
	return t.update(id, func(task TrackedTask, now time.Time) TrackedTask {
		task.Status = StatusDone
		task.Output = copyOutput(output)
		task.FinishedAt = &now
		task.UpdatedAt = now
		return task
	})
}

func (t *Tracker) Fail(id, message string) (TrackedTask, bool) {
	return t.update(id, func(task TrackedTask, now time.Time) TrackedTask {
		task.Status = StatusFailed
		task.Error = message
		task.FinishedAt = &now
		task.UpdatedAt = now
		return task
	})
}

func (t *Tracker) Get(id string) (TrackedTask, bool) {
	if t == nil {
		return TrackedTask{}, false
	}

	t.mu.RLock()
	defer t.mu.RUnlock()
	task, ok := t.tasks[id]
	if !ok {
		return TrackedTask{}, false
	}
	return copyTrackedTask(task), true
}

func (t *Tracker) update(id string, mutate func(TrackedTask, time.Time) TrackedTask) (TrackedTask, bool) {
	if t == nil {
		return TrackedTask{}, false
	}

	t.mu.Lock()
	defer t.mu.Unlock()
	task, ok := t.tasks[id]
	if !ok {
		return TrackedTask{}, false
	}
	task = mutate(task, t.currentTime())
	t.tasks[id] = task
	return copyTrackedTask(task), true
}

func (t *Tracker) currentTime() time.Time {
	if t.now == nil {
		return time.Now()
	}
	return t.now()
}

func copyTrackedTask(task TrackedTask) TrackedTask {
	task.Output = copyOutput(task.Output)
	return task
}

func copyOutput(output map[string]any) map[string]any {
	if len(output) == 0 {
		return nil
	}
	copied := make(map[string]any, len(output))
	for key, value := range output {
		copied[key] = value
	}
	return copied
}
