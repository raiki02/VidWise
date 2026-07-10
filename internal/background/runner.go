package background

import (
	"context"
	"log/slog"
	"runtime/debug"
	"sync"
	"time"
)

const defaultTimeout = 30 * time.Second

// Runner starts bounded background tasks that should outlive the request that
// scheduled them, such as memory extraction or post-extraction indexing.
type Runner struct {
	timeout time.Duration
	wg      sync.WaitGroup
}

// NewRunner creates a background task runner with a per-task timeout.
func NewRunner(timeout time.Duration) *Runner {
	if timeout <= 0 {
		timeout = defaultTimeout
	}
	return &Runner{timeout: timeout}
}

// Go starts fn in a goroutine with panic recovery and a detached timeout
// context. It returns false when there is no task to run.
func (r *Runner) Go(name string, fn func(context.Context)) bool {
	if fn == nil {
		return false
	}
	if r == nil {
		r = NewRunner(defaultTimeout)
	}
	if name == "" {
		name = "background.task"
	}

	r.wg.Add(1)
	go func() {
		defer r.wg.Done()
		defer func() {
			if recovered := recover(); recovered != nil {
				slog.Error("background.task_panic", "task", name, "panic", recovered, "stack", string(debug.Stack()))
			}
		}()

		ctx, cancel := context.WithTimeout(context.Background(), r.timeout)
		defer cancel()

		start := time.Now()
		fn(ctx)
		if err := ctx.Err(); err == context.DeadlineExceeded {
			slog.Warn("background.task_timeout", "task", name, "timeout", r.timeout)
			return
		}
		slog.Info("background.task_done", "task", name, "elapsed", time.Since(start))
	}()
	return true
}

// Wait blocks until all tasks scheduled before the wait complete or ctx is
// canceled. It is mainly useful for tests and graceful shutdown hooks.
func (r *Runner) Wait(ctx context.Context) error {
	if r == nil {
		return nil
	}

	done := make(chan struct{})
	go func() {
		r.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
