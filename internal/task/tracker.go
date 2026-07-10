package task

import (
	"sort"
	"sync"
	"time"
)

const (
	defaultMaxTrackedTasks = 1000
	defaultTaskRetention   = 24 * time.Hour
	defaultTaskListLimit   = 50
	maxTaskListLimit       = 100
)

// TrackedTask is the request-facing view of an asynchronous task.
type TrackedTask struct {
	ID         string         `json:"task_id"`
	Type       string         `json:"type"`
	Status     Status         `json:"status"`
	Steps      []TrackedStep  `json:"steps,omitempty"`
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

// TrackedStep is the request-facing view of one task pipeline step.
type TrackedStep struct {
	Name       string     `json:"name"`
	Status     Status     `json:"status"`
	Error      string     `json:"error,omitempty"`
	CreatedAt  time.Time  `json:"created_at"`
	UpdatedAt  time.Time  `json:"updated_at"`
	StartedAt  *time.Time `json:"started_at,omitempty"`
	FinishedAt *time.Time `json:"finished_at,omitempty"`
}

type TrackCreateRequest struct {
	ID        string
	Type      string
	UserID    string
	SessionID string
	TraceID   string
	Steps     []string
}

type TrackListRequest struct {
	UserID    string
	SessionID string
	Status    Status
	Limit     int
}

type TrackerOptions struct {
	MaxTasks  int
	RetainFor time.Duration
	Now       func() time.Time
}

// Tracker records in-process task state for async work launched by the gateway.
type Tracker struct {
	mu        sync.RWMutex
	tasks     map[string]TrackedTask
	now       func() time.Time
	maxTasks  int
	retainFor time.Duration
}

func NewTracker() *Tracker {
	return NewTrackerWithOptions(TrackerOptions{})
}

func NewTrackerWithOptions(opts TrackerOptions) *Tracker {
	opts = normalizeTrackerOptions(opts)
	return &Tracker{
		tasks:     make(map[string]TrackedTask),
		now:       opts.Now,
		maxTasks:  opts.MaxTasks,
		retainFor: opts.RetainFor,
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
		Steps:     newTrackedSteps(req.Steps, now),
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
	t.pruneLocked(now)
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

func (t *Tracker) StartStep(taskID, stepName string) (TrackedTask, bool) {
	return t.updateStep(taskID, stepName, func(step TrackedStep, now time.Time) TrackedStep {
		step.Status = StatusRunning
		step.StartedAt = &now
		step.UpdatedAt = now
		return step
	})
}

func (t *Tracker) CompleteStep(taskID, stepName string) (TrackedTask, bool) {
	return t.updateStep(taskID, stepName, func(step TrackedStep, now time.Time) TrackedStep {
		step.Status = StatusDone
		step.FinishedAt = &now
		step.UpdatedAt = now
		return step
	})
}

func (t *Tracker) FailStep(taskID, stepName, message string) (TrackedTask, bool) {
	return t.updateStep(taskID, stepName, func(step TrackedStep, now time.Time) TrackedStep {
		step.Status = StatusFailed
		step.Error = message
		step.FinishedAt = &now
		step.UpdatedAt = now
		return step
	})
}

func (t *Tracker) SkipStep(taskID, stepName, reason string) (TrackedTask, bool) {
	return t.updateStep(taskID, stepName, func(step TrackedStep, now time.Time) TrackedStep {
		step.Status = StatusSkipped
		step.Error = reason
		step.FinishedAt = &now
		step.UpdatedAt = now
		return step
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

func (t *Tracker) List(req TrackListRequest) []TrackedTask {
	if t == nil {
		return nil
	}

	now := t.currentTime()
	limit := normalizeTaskListLimit(req.Limit)

	t.mu.Lock()
	defer t.mu.Unlock()
	t.pruneLocked(now)

	matches := make([]TrackedTask, 0, len(t.tasks))
	for _, task := range t.tasks {
		if req.UserID != "" && task.UserID != req.UserID {
			continue
		}
		if req.SessionID != "" && task.SessionID != req.SessionID {
			continue
		}
		if req.Status != "" && task.Status != req.Status {
			continue
		}
		matches = append(matches, copyTrackedTask(task))
	}
	sort.Slice(matches, func(i, j int) bool {
		if matches[i].UpdatedAt.Equal(matches[j].UpdatedAt) {
			return matches[i].ID < matches[j].ID
		}
		return matches[i].UpdatedAt.After(matches[j].UpdatedAt)
	})
	if len(matches) > limit {
		matches = matches[:limit]
	}
	return matches
}

func (t *Tracker) updateStep(taskID, stepName string, mutate func(TrackedStep, time.Time) TrackedStep) (TrackedTask, bool) {
	if t == nil {
		return TrackedTask{}, false
	}

	t.mu.Lock()
	defer t.mu.Unlock()
	task, ok := t.tasks[taskID]
	if !ok {
		return TrackedTask{}, false
	}
	idx := findTrackedStep(task.Steps, stepName)
	if idx < 0 {
		return TrackedTask{}, false
	}
	now := t.currentTime()
	task.Steps[idx] = mutate(task.Steps[idx], now)
	task.UpdatedAt = now
	t.tasks[taskID] = task
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

func (t *Tracker) Prune() int {
	if t == nil {
		return 0
	}

	now := t.currentTime()
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.pruneLocked(now)
}

func (t *Tracker) currentTime() time.Time {
	if t.now == nil {
		return time.Now()
	}
	return t.now()
}

func (t *Tracker) pruneLocked(now time.Time) int {
	if len(t.tasks) == 0 {
		return 0
	}
	t.normalizeOptionsLocked()

	removed := 0
	if t.retainFor > 0 {
		for id, task := range t.tasks {
			if !isTerminalTaskStatus(task.Status) {
				continue
			}
			if now.Sub(taskTerminalTime(task)) >= t.retainFor {
				delete(t.tasks, id)
				removed++
			}
		}
	}

	if t.maxTasks > 0 && len(t.tasks) > t.maxTasks {
		candidates := trackedTaskPruneCandidates(t.tasks)
		for _, candidate := range candidates {
			if len(t.tasks) <= t.maxTasks {
				break
			}
			delete(t.tasks, candidate.id)
			removed++
		}
	}

	return removed
}

func (t *Tracker) normalizeOptionsLocked() {
	if t.maxTasks <= 0 {
		t.maxTasks = defaultMaxTrackedTasks
	}
	if t.retainFor <= 0 {
		t.retainFor = defaultTaskRetention
	}
}

func copyTrackedTask(task TrackedTask) TrackedTask {
	task.Output = copyOutput(task.Output)
	if len(task.Steps) > 0 {
		steps := make([]TrackedStep, len(task.Steps))
		copy(steps, task.Steps)
		task.Steps = steps
	}
	return task
}

func newTrackedSteps(names []string, now time.Time) []TrackedStep {
	if len(names) == 0 {
		return nil
	}
	steps := make([]TrackedStep, 0, len(names))
	seen := make(map[string]struct{}, len(names))
	for _, name := range names {
		if name == "" {
			continue
		}
		if _, ok := seen[name]; ok {
			continue
		}
		seen[name] = struct{}{}
		steps = append(steps, TrackedStep{
			Name:      name,
			Status:    StatusPending,
			CreatedAt: now,
			UpdatedAt: now,
		})
	}
	return steps
}

func findTrackedStep(steps []TrackedStep, name string) int {
	for i, step := range steps {
		if step.Name == name {
			return i
		}
	}
	return -1
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

func normalizeTrackerOptions(opts TrackerOptions) TrackerOptions {
	if opts.MaxTasks <= 0 {
		opts.MaxTasks = defaultMaxTrackedTasks
	}
	if opts.RetainFor <= 0 {
		opts.RetainFor = defaultTaskRetention
	}
	if opts.Now == nil {
		opts.Now = time.Now
	}
	return opts
}

func normalizeTaskListLimit(limit int) int {
	if limit <= 0 {
		return defaultTaskListLimit
	}
	if limit > maxTaskListLimit {
		return maxTaskListLimit
	}
	return limit
}

type trackedTaskPruneCandidate struct {
	id   string
	task TrackedTask
}

func trackedTaskPruneCandidates(tasks map[string]TrackedTask) []trackedTaskPruneCandidate {
	candidates := make([]trackedTaskPruneCandidate, 0, len(tasks))
	for id, task := range tasks {
		if !isTerminalTaskStatus(task.Status) {
			continue
		}
		candidates = append(candidates, trackedTaskPruneCandidate{
			id:   id,
			task: task,
		})
	}
	sort.Slice(candidates, func(i, j int) bool {
		left := taskTerminalTime(candidates[i].task)
		right := taskTerminalTime(candidates[j].task)
		if left.Equal(right) {
			return candidates[i].id < candidates[j].id
		}
		return left.Before(right)
	})
	return candidates
}

func isTerminalTaskStatus(status Status) bool {
	return status == StatusDone || status == StatusFailed
}

func taskTerminalTime(task TrackedTask) time.Time {
	if task.FinishedAt != nil {
		return *task.FinishedAt
	}
	if !task.UpdatedAt.IsZero() {
		return task.UpdatedAt
	}
	return task.CreatedAt
}
