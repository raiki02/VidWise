package task

import (
	"testing"
	"time"
)

func TestTrackerRecordsTaskLifecycle(t *testing.T) {
	now := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	tracker := NewTracker()
	tracker.now = func() time.Time {
		now = now.Add(time.Second)
		return now
	}

	created := tracker.Create(TrackCreateRequest{
		ID:        "task-1",
		Type:      "video_process",
		UserID:    "u1",
		SessionID: "s1",
		TraceID:   "trace-1",
	})
	if created.Status != StatusPending {
		t.Fatalf("created status = %q, want pending", created.Status)
	}

	running, ok := tracker.Start("task-1")
	if !ok {
		t.Fatal("expected task to start")
	}
	if running.Status != StatusRunning || running.StartedAt == nil {
		t.Fatalf("running task = %#v, want running with started_at", running)
	}

	done, ok := tracker.Complete("task-1", map[string]any{"text_length": 42})
	if !ok {
		t.Fatal("expected task to complete")
	}
	if done.Status != StatusDone || done.FinishedAt == nil {
		t.Fatalf("done task = %#v, want done with finished_at", done)
	}
	if done.Output["text_length"] != 42 {
		t.Fatalf("output = %#v, want text_length", done.Output)
	}

	got, ok := tracker.Get("task-1")
	if !ok {
		t.Fatal("expected task to be queryable")
	}
	if got.Status != StatusDone || got.TraceID != "trace-1" {
		t.Fatalf("queried task = %#v", got)
	}
}

func TestTrackerFailsTask(t *testing.T) {
	tracker := NewTracker()
	tracker.Create(TrackCreateRequest{ID: "task-1"})

	got, ok := tracker.Fail("task-1", "pipeline failed")
	if !ok {
		t.Fatal("expected task to fail")
	}
	if got.Status != StatusFailed {
		t.Fatalf("status = %q, want failed", got.Status)
	}
	if got.Error != "pipeline failed" {
		t.Fatalf("error = %q, want pipeline failed", got.Error)
	}
}

func TestTrackerReturnsCopies(t *testing.T) {
	tracker := NewTracker()
	tracker.Create(TrackCreateRequest{ID: "task-1"})
	got, ok := tracker.Complete("task-1", map[string]any{"text_length": 42})
	if !ok {
		t.Fatal("expected task to complete")
	}
	got.Output["text_length"] = 0

	again, ok := tracker.Get("task-1")
	if !ok {
		t.Fatal("expected task to be queryable")
	}
	if again.Output["text_length"] != 42 {
		t.Fatalf("tracker leaked mutable output map: %#v", again.Output)
	}
}

func TestTrackerReportsMissingTask(t *testing.T) {
	tracker := NewTracker()
	if _, ok := tracker.Get("missing"); ok {
		t.Fatal("expected missing task")
	}
	if _, ok := tracker.Start("missing"); ok {
		t.Fatal("expected missing task on start")
	}
}

func TestNilTrackerDoesNotPretendToCreateTask(t *testing.T) {
	var tracker *Tracker
	got := tracker.Create(TrackCreateRequest{ID: "task-1"})
	if got.ID != "" {
		t.Fatalf("nil tracker created task: %#v", got)
	}
	if _, ok := tracker.Get("task-1"); ok {
		t.Fatal("expected nil tracker to report missing task")
	}
}

func TestZeroValueTrackerIsUsable(t *testing.T) {
	var tracker Tracker
	created := tracker.Create(TrackCreateRequest{ID: "task-1"})
	if created.ID != "task-1" {
		t.Fatalf("created task = %#v", created)
	}

	got, ok := tracker.Get("task-1")
	if !ok {
		t.Fatal("expected zero-value tracker to store task")
	}
	if got.Status != StatusPending {
		t.Fatalf("status = %q, want pending", got.Status)
	}
}
