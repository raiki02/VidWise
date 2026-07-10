package background

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestRunnerRunsTaskAndWaits(t *testing.T) {
	runner := NewRunner(time.Second)
	done := make(chan struct{})

	if ok := runner.Go("test.done", func(context.Context) {
		close(done)
	}); !ok {
		t.Fatal("expected task to be scheduled")
	}

	if err := runner.Wait(context.Background()); err != nil {
		t.Fatalf("wait: %v", err)
	}

	select {
	case <-done:
	default:
		t.Fatal("expected task to run")
	}
}

func TestRunnerRecoversPanics(t *testing.T) {
	runner := NewRunner(time.Second)

	if ok := runner.Go("test.panic", func(context.Context) {
		panic("boom")
	}); !ok {
		t.Fatal("expected task to be scheduled")
	}

	if err := runner.Wait(context.Background()); err != nil {
		t.Fatalf("wait: %v", err)
	}
}

func TestRunnerPassesTimeoutContext(t *testing.T) {
	runner := NewRunner(10 * time.Millisecond)
	errs := make(chan error, 1)

	if ok := runner.Go("test.timeout", func(ctx context.Context) {
		<-ctx.Done()
		errs <- ctx.Err()
	}); !ok {
		t.Fatal("expected task to be scheduled")
	}

	waitCtx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := runner.Wait(waitCtx); err != nil {
		t.Fatalf("wait: %v", err)
	}

	if err := <-errs; !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("context err = %v, want deadline exceeded", err)
	}
}

func TestRunnerRejectsNilTask(t *testing.T) {
	runner := NewRunner(time.Second)
	if ok := runner.Go("test.nil", nil); ok {
		t.Fatal("expected nil task to be rejected")
	}
}
