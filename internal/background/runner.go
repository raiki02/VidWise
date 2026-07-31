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
	timeout       time.Duration
	maxConcurrent int
	slots         chan struct{}
	wg            sync.WaitGroup
}

// Options configures a Runner.
type Options struct {
	Timeout       time.Duration
	MaxConcurrent int
}

// NewRunner creates a background task runner with a per-task timeout.
func NewRunner(timeout time.Duration) *Runner {
	return NewRunnerWithOptions(Options{Timeout: timeout})
}

// NewRunnerWithOptions creates a background task runner with optional
// concurrency limiting. A non-positive MaxConcurrent leaves concurrency
// unlimited.
func NewRunnerWithOptions(opts Options) *Runner {
	if opts.Timeout <= 0 {
		opts.Timeout = defaultTimeout
	}
	if opts.MaxConcurrent < 0 {
		opts.MaxConcurrent = 0
	}

	r := &Runner{timeout: opts.Timeout, maxConcurrent: opts.MaxConcurrent}
	if opts.MaxConcurrent > 0 {
		r.slots = make(chan struct{}, opts.MaxConcurrent)
	}
	return r
}

// Go starts fn in a goroutine with panic recovery and a detached timeout
// context. It returns false when there is no task to run or the concurrency
// limit is already full.
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

	var release func()
	if r.slots != nil {
		select {
		case r.slots <- struct{}{}:
			release = func() {
				<-r.slots
			}
		default:
			slog.Warn("background.task_rejected", "task", name, "max_concurrent", r.maxConcurrent)
			return false
		}
	}

	r.wg.Add(1)
	go func() {
		defer r.wg.Done()
		if release != nil {
			defer release()
		}
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
