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

func TestRunnerRejectsWhenConcurrencyLimitIsFull(t *testing.T) {
	runner := NewRunnerWithOptions(Options{
		Timeout:       time.Second,
		MaxConcurrent: 1,
	})
	started := make(chan struct{})
	release := make(chan struct{})
	rejectedRan := make(chan struct{}, 1)

	if ok := runner.Go("test.blocking", func(context.Context) {
		close(started)
		<-release
	}); !ok {
		t.Fatal("expected first task to be scheduled")
	}

	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for first task")
	}

	if ok := runner.Go("test.rejected", func(context.Context) {
		rejectedRan <- struct{}{}
	}); ok {
		t.Fatal("expected second task to be rejected while limit is full")
	}

	close(release)
	if err := runner.Wait(context.Background()); err != nil {
		t.Fatalf("wait: %v", err)
	}

	select {
	case <-rejectedRan:
		t.Fatal("rejected task should not run")
	default:
	}
}

func TestRunnerAllowsNewTaskAfterLimitedTaskCompletes(t *testing.T) {
	runner := NewRunnerWithOptions(Options{
		Timeout:       time.Second,
		MaxConcurrent: 1,
	})
	firstDone := make(chan struct{})
	secondDone := make(chan struct{})

	if ok := runner.Go("test.first", func(context.Context) {
		close(firstDone)
	}); !ok {
		t.Fatal("expected first task to be scheduled")
	}
	if err := runner.Wait(context.Background()); err != nil {
		t.Fatalf("wait first: %v", err)
	}

	if ok := runner.Go("test.second", func(context.Context) {
		close(secondDone)
	}); !ok {
		t.Fatal("expected second task to be scheduled after first completed")
	}
	if err := runner.Wait(context.Background()); err != nil {
		t.Fatalf("wait second: %v", err)
	}

	select {
	case <-firstDone:
	default:
		t.Fatal("expected first task to run")
	}
	select {
	case <-secondDone:
	default:
		t.Fatal("expected second task to run")
	}
}
