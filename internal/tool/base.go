package tool

import (
	"context"
	"encoding/json"
	"log/slog"
	"time"

	"github.com/cloudwego/eino/components/tool"
	"github.com/sony/gobreaker"
)

// Wrapper wraps an Eino InvokableTool with retry, circuit breaker,
// timeout, trace_id propagation, and execution logging.
type Wrapper struct {
	inner   tool.InvokableTool
	name    string
	cb      *gobreaker.CircuitBreaker
	timeout time.Duration
}

type WrapperConfig struct {
	Name          string
	MaxRetry      int
	Timeout       time.Duration
	CBMaxRequests uint32
	CBInterval    time.Duration
	CBTimeout     time.Duration
}

func NewWrapper(inner tool.InvokableTool, cfg WrapperConfig) *Wrapper {
	if cfg.MaxRetry <= 0 {
		cfg.MaxRetry = 3
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 5 * time.Minute
	}
	if cfg.CBMaxRequests == 0 {
		cfg.CBMaxRequests = 5
	}
	if cfg.CBInterval == 0 {
		cfg.CBInterval = 60 * time.Second
	}
	if cfg.CBTimeout == 0 {
		cfg.CBTimeout = 30 * time.Second
	}

	cb := gobreaker.NewCircuitBreaker(gobreaker.Settings{
		Name:        cfg.Name,
		MaxRequests: cfg.CBMaxRequests,
		Interval:    cfg.CBInterval,
		Timeout:     cfg.CBTimeout,
		ReadyToTrip: func(counts gobreaker.Counts) bool {
			return counts.ConsecutiveFailures > 3
		},
	})

	return &Wrapper{
		inner:   inner,
		name:    cfg.Name,
		cb:      cb,
		timeout: cfg.Timeout,
	}
}

// Run executes the tool with retry, timeout, and circuit breaker.
func (w *Wrapper) Run(ctx context.Context, args string) (string, error) {
	start := time.Now()

	result, err := w.cb.Execute(func() (interface{}, error) {
		var lastErr error
		for attempt := 0; attempt <= 3; attempt++ {
			toolCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), w.timeout)
			defer cancel()

			output, err := w.inner.InvokableRun(toolCtx, args)
			if err == nil {
				return output, nil
			}
			lastErr = err
			slog.Warn("tool.retry", "tool", w.name, "attempt", attempt+1, "err", err)
			time.Sleep(time.Duration(1<<attempt) * time.Second)
		}
		return nil, lastErr
	})

	elapsed := time.Since(start)
	if err != nil {
		slog.Error("tool.failed", "tool", w.name, "elapsed", elapsed, "err", err)
		return "", err
	}
	output, _ := result.(string)
	slog.Info("tool.done", "tool", w.name, "elapsed", elapsed)
	return output, nil
}

// InvokableRun satisfies the tool.InvokableTool interface
// so the wrapper itself can be used as an invokable tool.
func (w *Wrapper) InvokableRun(ctx context.Context, args string, opts ...tool.Option) (string, error) {
	return w.Run(ctx, args)
}

// Name returns the tool name.
func (w *Wrapper) Name() string { return w.name }

// ToJSON is a helper to encode arguments and return the JSON string.
func ToJSON(v any) (string, error) {
	b, err := json.Marshal(v)
	return string(b), err
}
