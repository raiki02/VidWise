package task

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"time"
)

// StepFunc is a function that executes a single step and returns its output.
type StepFunc func(step *Step, prevOutputs map[string]json.RawMessage) (any, error)

// Runner executes task step DAGs with topological ordering.
type Runner struct {
	repo    *Repo
	manager *Manager
	steps   map[string]StepFunc
}

func NewRunner(repo *Repo, manager *Manager) *Runner {
	return &Runner{
		repo:    repo,
		manager: manager,
		steps:   make(map[string]StepFunc),
	}
}

// RegisterStep registers a step executor for a given step name.
func (r *Runner) RegisterStep(name string, fn StepFunc) {
	r.steps[name] = fn
}

// Run executes all steps in DAG order for a task.
func (r *Runner) Run(taskID string) error {
	_, steps, err := r.manager.GetTaskWithSteps(nil, taskID)
	if err != nil {
		return fmt.Errorf("get task: %w", err)
	}

	if err := r.manager.BeginTask(nil, taskID); err != nil {
		return fmt.Errorf("begin task: %w", err)
	}

	slog.Info("task.runner.start", "task_id", taskID)

	prevOutputs := make(map[string]json.RawMessage)
	done := make(map[string]bool)

	for len(done) < len(steps) {
		var ready []*Step
	stepLoop:
		for _, s := range steps {
			if done[s.Name] || s.Status == StatusRunning {
				continue
			}

			if len(s.DependsOn) > 0 {
				var deps []string
				if err := json.Unmarshal(s.DependsOn, &deps); err == nil {
					for _, dep := range deps {
						if !done[dep] {
							continue stepLoop
						}
					}
				}
			}
			ready = append(ready, s)
		}

		if len(ready) == 0 {
			for _, s := range steps {
				if !done[s.Name] && s.Status == StatusFailed {
					_ = r.manager.FailTask(nil, taskID, fmt.Sprintf("step %s failed", s.Name))
					return fmt.Errorf("step %s failed", s.Name)
				}
			}
			break
		}

		for _, step := range ready {
			if err := r.executeStep(step, prevOutputs); err != nil {
				slog.Error("task.runner.step_failed", "task_id", taskID, "step", step.Name, "err", err)
				return fmt.Errorf("step %s: %w", step.Name, err)
			}
			done[step.Name] = true
			if step.Output != nil {
				prevOutputs[step.Name] = step.Output
			}
		}
	}

	slog.Info("task.runner.done", "task_id", taskID)
	_ = r.manager.CompleteTask(nil, taskID, prevOutputs)
	return nil
}

func (r *Runner) executeStep(step *Step, prevOutputs map[string]json.RawMessage) error {
	slog.Info("task.runner.step_start", "step", step.Name, "step_id", step.ID)

	if err := r.repo.UpdateStepStatus(nil, step.ID, StatusRunning, nil, ""); err != nil {
		return fmt.Errorf("mark step running: %w", err)
	}

	start := time.Now()
	fn, ok := r.steps[step.Name]
	if !ok {
		return fmt.Errorf("no executor registered for step: %s", step.Name)
	}

	output, err := fn(step, prevOutputs)
	if err != nil {
		_ = r.repo.UpdateStepStatus(nil, step.ID, StatusFailed, nil, err.Error())
		return err
	}

	var outputJSON json.RawMessage
	if output != nil {
		outputJSON, _ = json.Marshal(output)
	}

	if err := r.repo.UpdateStepStatus(nil, step.ID, StatusDone, outputJSON, ""); err != nil {
		return fmt.Errorf("mark step done: %w", err)
	}
	slog.Info("task.runner.step_done", "step", step.Name, "elapsed", time.Since(start))
	return nil
}
