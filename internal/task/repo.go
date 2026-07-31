package task

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	mysqlclient "github.com/raiki02/vidwise/internal/storage/mysql"
	"gorm.io/gorm"
)

type Repo struct {
	db *gorm.DB
}

func NewRepo(client *mysqlclient.Client) *Repo {
	if client == nil {
		return nil
	}
	return NewRepoFromDB(client.DB)
}

func NewRepoFromDB(db *gorm.DB) *Repo {
	if db == nil {
		return nil
	}
	return &Repo{db: db}
}

func (r *Repo) AutoMigrate() error {
	if r == nil || r.db == nil {
		return errors.New("task repo is unavailable")
	}
	return r.db.AutoMigrate(&Task{}, &Step{}, &Transcript{}, &ToolLog{})
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

// Load restores tracked task state for the gateway task tracker.
func (r *Repo) Load(ctx context.Context) ([]TrackedTask, error) {
	if r == nil || r.db == nil {
		return nil, errors.New("task repo is unavailable")
	}

	var tasks []Task
	if err := r.db.WithContext(ctx).Order("updated_at DESC").Find(&tasks).Error; err != nil {
		return nil, fmt.Errorf("load tracked tasks: %w", err)
	}
	if len(tasks) == 0 {
		return nil, nil
	}

	taskIDs := make([]string, 0, len(tasks))
	for _, task := range tasks {
		if strings.TrimSpace(task.ID) != "" {
			taskIDs = append(taskIDs, task.ID)
		}
	}

	stepsByTask, err := r.loadTrackedSteps(ctx, taskIDs)
	if err != nil {
		return nil, err
	}
	transcriptsByTask, err := r.loadTrackedTranscripts(ctx, taskIDs)
	if err != nil {
		return nil, err
	}

	out := make([]TrackedTask, 0, len(tasks))
	for _, task := range tasks {
		out = append(out, trackedTaskFromRecord(task, stepsByTask[task.ID], transcriptsByTask[task.ID]))
	}
	return out, nil
}

// SaveTask persists one tracked task and its current step/transcript view.
func (r *Repo) SaveTask(ctx context.Context, task TrackedTask) error {
	if r == nil || r.db == nil {
		return errors.New("task repo is unavailable")
	}
	if strings.TrimSpace(task.ID) == "" {
		return nil
	}

	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		record, err := taskRecordFromTracked(task)
		if err != nil {
			return err
		}
		if err := tx.Save(&record).Error; err != nil {
			return fmt.Errorf("save tracked task: %w", err)
		}
		if err := saveTrackedSteps(tx, task); err != nil {
			return err
		}
		if err := saveTrackedTranscript(tx, task); err != nil {
			return err
		}
		return nil
	})
}

// DeleteTasks removes pruned tracked task records from MySQL.
func (r *Repo) DeleteTasks(ctx context.Context, ids []string) error {
	if r == nil || r.db == nil || len(ids) == 0 {
		return nil
	}
	ids = normalizeIDs(ids)
	if len(ids) == 0 {
		return nil
	}

	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		if err := tx.Where("task_id IN ?", ids).Delete(&Step{}).Error; err != nil {
			return fmt.Errorf("delete task steps: %w", err)
		}
		if err := tx.Where("task_id IN ?", ids).Delete(&Transcript{}).Error; err != nil {
			return fmt.Errorf("delete transcripts: %w", err)
		}
		if err := tx.Where("task_id IN ?", ids).Delete(&ToolLog{}).Error; err != nil {
			return fmt.Errorf("delete tool logs: %w", err)
		}
		if err := tx.Where("id IN ?", ids).Delete(&Task{}).Error; err != nil {
			return fmt.Errorf("delete tasks: %w", err)
		}
		return nil
	})
}

func (r *Repo) loadTrackedSteps(ctx context.Context, taskIDs []string) (map[string][]Step, error) {
	out := make(map[string][]Step)
	if len(taskIDs) == 0 {
		return out, nil
	}

	var steps []Step
	if err := r.db.WithContext(ctx).
		Where("task_id IN ?", taskIDs).
		Order("created_at ASC, name ASC").
		Find(&steps).Error; err != nil {
		return nil, fmt.Errorf("load tracked steps: %w", err)
	}
	for _, step := range steps {
		out[step.TaskID] = append(out[step.TaskID], step)
	}
	return out, nil
}

func (r *Repo) loadTrackedTranscripts(ctx context.Context, taskIDs []string) (map[string]Transcript, error) {
	out := make(map[string]Transcript)
	if len(taskIDs) == 0 {
		return out, nil
	}

	var transcripts []Transcript
	if err := r.db.WithContext(ctx).
		Where("task_id IN ?", taskIDs).
		Order("created_at DESC").
		Find(&transcripts).Error; err != nil {
		return nil, fmt.Errorf("load tracked transcripts: %w", err)
	}
	for _, transcript := range transcripts {
		if _, exists := out[transcript.TaskID]; exists {
			continue
		}
		out[transcript.TaskID] = transcript
	}
	return out, nil
}

func taskRecordFromTracked(task TrackedTask) (Task, error) {
	outputJSON, err := outputJSON(task.Output)
	if err != nil {
		return Task{}, err
	}
	status := task.Status
	if status == "" {
		status = StatusPending
	}
	createdAt := task.CreatedAt
	if createdAt.IsZero() {
		createdAt = time.Now()
	}
	updatedAt := task.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = createdAt
	}
	return Task{
		ID:         task.ID,
		UserID:     task.UserID,
		SessionID:  task.SessionID,
		Type:       task.Type,
		Status:     status,
		Output:     outputJSON,
		RetryCount: 0,
		MaxRetries: 3,
		TraceID:    stringPtr(task.TraceID),
		ErrorMsg:   stringPtr(task.Error),
		CreatedAt:  createdAt,
		UpdatedAt:  updatedAt,
		StartedAt:  task.StartedAt,
		FinishedAt: task.FinishedAt,
	}, nil
}

func trackedTaskFromRecord(task Task, steps []Step, transcript Transcript) TrackedTask {
	output := outputMap(task.Output)
	output = patchOutputFromTranscript(output, transcript)
	return TrackedTask{
		ID:         task.ID,
		Type:       task.Type,
		Status:     task.Status,
		Steps:      trackedStepsFromRecords(steps),
		UserID:     task.UserID,
		SessionID:  task.SessionID,
		TraceID:    stringFromPtr(task.TraceID),
		Error:      stringFromPtr(task.ErrorMsg),
		Output:     output,
		CreatedAt:  task.CreatedAt,
		UpdatedAt:  task.UpdatedAt,
		StartedAt:  task.StartedAt,
		FinishedAt: task.FinishedAt,
	}
}

func trackedStepsFromRecords(steps []Step) []TrackedStep {
	if len(steps) == 0 {
		return nil
	}
	out := make([]TrackedStep, 0, len(steps))
	for _, step := range steps {
		out = append(out, TrackedStep{
			Name:       step.Name,
			Status:     step.Status,
			Error:      stringFromPtr(step.ErrorMsg),
			CreatedAt:  step.CreatedAt,
			UpdatedAt:  step.UpdatedAt,
			StartedAt:  step.StartedAt,
			FinishedAt: step.FinishedAt,
		})
	}
	return out
}

func saveTrackedSteps(tx *gorm.DB, task TrackedTask) error {
	names := make([]string, 0, len(task.Steps))
	for _, tracked := range task.Steps {
		name := strings.TrimSpace(tracked.Name)
		if name == "" {
			continue
		}
		names = append(names, name)

		createdAt := tracked.CreatedAt
		if createdAt.IsZero() {
			createdAt = firstNonZeroTime(task.CreatedAt, task.UpdatedAt, time.Now())
		}
		updatedAt := tracked.UpdatedAt
		if updatedAt.IsZero() {
			updatedAt = firstNonZeroTime(task.UpdatedAt, createdAt, time.Now())
		}
		updates := map[string]any{
			"status":      nonEmptyStatus(tracked.Status, StatusPending),
			"error_msg":   stringPtr(tracked.Error),
			"created_at":  createdAt,
			"updated_at":  updatedAt,
			"started_at":  tracked.StartedAt,
			"finished_at": tracked.FinishedAt,
		}

		var existing Step
		err := tx.Where("task_id = ? AND name = ?", task.ID, name).First(&existing).Error
		if errors.Is(err, gorm.ErrRecordNotFound) {
			step := Step{
				TaskID:     task.ID,
				Name:       name,
				Status:     nonEmptyStatus(tracked.Status, StatusPending),
				ErrorMsg:   stringPtr(tracked.Error),
				CreatedAt:  createdAt,
				UpdatedAt:  updatedAt,
				StartedAt:  tracked.StartedAt,
				FinishedAt: tracked.FinishedAt,
			}
			if err := tx.Create(&step).Error; err != nil {
				return fmt.Errorf("create tracked step %s: %w", name, err)
			}
			continue
		}
		if err != nil {
			return fmt.Errorf("lookup tracked step %s: %w", name, err)
		}
		if err := tx.Model(&Step{}).Where("id = ?", existing.ID).Updates(updates).Error; err != nil {
			return fmt.Errorf("update tracked step %s: %w", name, err)
		}
	}

	if len(names) == 0 {
		return tx.Where("task_id = ?", task.ID).Delete(&Step{}).Error
	}
	return tx.Where("task_id = ? AND name NOT IN ?", task.ID, names).Delete(&Step{}).Error
}

func saveTrackedTranscript(tx *gorm.DB, task TrackedTask) error {
	rawText, formattedText := transcriptTexts(task.Output)
	if rawText == "" && formattedText == "" {
		return nil
	}

	updates := map[string]any{
		"session_id":     task.SessionID,
		"user_id":        task.UserID,
		"raw_text":       stringPtr(rawText),
		"formatted_text": stringPtr(formattedText),
	}

	var existing Transcript
	err := tx.Where("task_id = ?", task.ID).Order("created_at DESC").First(&existing).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		transcript := Transcript{
			TaskID:        task.ID,
			SessionID:     task.SessionID,
			UserID:        task.UserID,
			RawText:       stringPtr(rawText),
			FormattedText: stringPtr(formattedText),
			CreatedAt:     firstNonZeroTime(task.CreatedAt, time.Now()),
		}
		if err := tx.Create(&transcript).Error; err != nil {
			return fmt.Errorf("create tracked transcript: %w", err)
		}
		return nil
	}
	if err != nil {
		return fmt.Errorf("lookup tracked transcript: %w", err)
	}
	if err := tx.Model(&Transcript{}).Where("id = ?", existing.ID).Updates(updates).Error; err != nil {
		return fmt.Errorf("update tracked transcript: %w", err)
	}
	return nil
}

func outputJSON(output map[string]any) (json.RawMessage, error) {
	if len(output) == 0 {
		return nil, nil
	}
	data, err := json.Marshal(output)
	if err != nil {
		return nil, fmt.Errorf("encode tracked task output: %w", err)
	}
	return data, nil
}

func outputMap(raw json.RawMessage) map[string]any {
	if len(raw) == 0 {
		return nil
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil
	}
	return out
}

func patchOutputFromTranscript(output map[string]any, transcript Transcript) map[string]any {
	rawText := stringFromPtr(transcript.RawText)
	formattedText := stringFromPtr(transcript.FormattedText)
	if rawText == "" && formattedText == "" {
		return output
	}
	if output == nil {
		output = make(map[string]any)
	}
	if rawText != "" {
		setOutputDefault(output, "raw_text", rawText)
		setOutputDefault(output, "text", rawText)
	}
	if formattedText != "" {
		setOutputDefault(output, "formatted_text", formattedText)
		if output["text"] == nil || output["text"] == rawText {
			output["text"] = formattedText
		}
		setOutputDefault(output, "text_formatted", true)
		setOutputDefault(output, "can_index_knowledge", true)
		setOutputDefault(output, "formatted_text_length", len([]rune(formattedText)))
	}
	return output
}

func transcriptTexts(output map[string]any) (string, string) {
	if len(output) == 0 {
		return "", ""
	}
	rawText := outputString(output, "raw_text")
	if rawText == "" {
		rawText = outputString(output, "text")
	}
	return rawText, outputString(output, "formatted_text")
}

func setOutputDefault(output map[string]any, key string, value any) {
	if _, ok := output[key]; ok {
		return
	}
	output[key] = value
}

func outputString(output map[string]any, key string) string {
	value, ok := output[key]
	if !ok {
		return ""
	}
	got, ok := value.(string)
	if !ok {
		return ""
	}
	return got
}

func normalizeIDs(ids []string) []string {
	out := make([]string, 0, len(ids))
	seen := map[string]struct{}{}
	for _, id := range ids {
		id = strings.TrimSpace(id)
		if id == "" {
			continue
		}
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		out = append(out, id)
	}
	return out
}

func stringPtr(value string) *string {
	if value == "" {
		return nil
	}
	return &value
}

func stringFromPtr(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func nonEmptyStatus(status, fallback Status) Status {
	if status == "" {
		return fallback
	}
	return status
}

func firstNonZeroTime(values ...time.Time) time.Time {
	for _, value := range values {
		if !value.IsZero() {
			return value
		}
	}
	return time.Time{}
}
