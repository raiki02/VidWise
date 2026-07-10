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
		Steps:     []string{"download_audio", "transcribe_audio"},
	})
	if created.Status != StatusPending {
		t.Fatalf("created status = %q, want pending", created.Status)
	}
	if len(created.Steps) != 2 || created.Steps[0].Status != StatusPending {
		t.Fatalf("created steps = %#v", created.Steps)
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

func TestTrackerRecordsStepLifecycle(t *testing.T) {
	tracker := NewTracker()
	tracker.Create(TrackCreateRequest{
		ID:    "task-1",
		Steps: []string{"download_audio", "transcribe_audio", "download_audio"},
	})

	got, ok := tracker.StartStep("task-1", "download_audio")
	if !ok {
		t.Fatal("expected step to start")
	}
	if len(got.Steps) != 2 {
		t.Fatalf("steps = %#v, want deduped steps", got.Steps)
	}
	if got.Steps[0].Status != StatusRunning || got.Steps[0].StartedAt == nil {
		t.Fatalf("started step = %#v", got.Steps[0])
	}

	got, ok = tracker.CompleteStep("task-1", "download_audio")
	if !ok {
		t.Fatal("expected step to complete")
	}
	if got.Steps[0].Status != StatusDone || got.Steps[0].FinishedAt == nil {
		t.Fatalf("completed step = %#v", got.Steps[0])
	}

	got, ok = tracker.FailStep("task-1", "transcribe_audio", "asr down")
	if !ok {
		t.Fatal("expected step to fail")
	}
	if got.Steps[1].Status != StatusFailed || got.Steps[1].Error != "asr down" {
		t.Fatalf("failed step = %#v", got.Steps[1])
	}
}

func TestTrackerSkipsStep(t *testing.T) {
	tracker := NewTracker()
	tracker.Create(TrackCreateRequest{ID: "task-1", Steps: []string{"format_transcript"}})

	got, ok := tracker.SkipStep("task-1", "format_transcript", "tool unavailable")
	if !ok {
		t.Fatal("expected step to be skipped")
	}
	if got.Steps[0].Status != StatusSkipped {
		t.Fatalf("status = %q, want skipped", got.Steps[0].Status)
	}
	if got.Steps[0].Error != "tool unavailable" {
		t.Fatalf("skip reason = %q, want tool unavailable", got.Steps[0].Error)
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
	tracker.Create(TrackCreateRequest{ID: "task-1", Steps: []string{"download_audio"}})
	got, ok := tracker.Complete("task-1", map[string]any{"text_length": 42})
	if !ok {
		t.Fatal("expected task to complete")
	}
	got.Output["text_length"] = 0
	got.Steps[0].Status = StatusFailed

	again, ok := tracker.Get("task-1")
	if !ok {
		t.Fatal("expected task to be queryable")
	}
	if again.Output["text_length"] != 42 {
		t.Fatalf("tracker leaked mutable output map: %#v", again.Output)
	}
	if again.Steps[0].Status != StatusPending {
		t.Fatalf("tracker leaked mutable steps: %#v", again.Steps)
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
	if _, ok := tracker.StartStep("missing", "download_audio"); ok {
		t.Fatal("expected missing task on step start")
	}
	tracker.Create(TrackCreateRequest{ID: "task-1", Steps: []string{"download_audio"}})
	if _, ok := tracker.StartStep("task-1", "missing"); ok {
		t.Fatal("expected missing step")
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

func TestTrackerPrunesExpiredTerminalTasks(t *testing.T) {
	now := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	tracker := NewTrackerWithOptions(TrackerOptions{
		MaxTasks:  10,
		RetainFor: time.Minute,
		Now: func() time.Time {
			return now
		},
	})

	tracker.Create(TrackCreateRequest{ID: "old-done"})
	tracker.Complete("old-done", nil)
	tracker.Create(TrackCreateRequest{ID: "old-running"})
	tracker.Start("old-running")

	now = now.Add(time.Minute + time.Second)
	removed := tracker.Prune()
	if removed != 1 {
		t.Fatalf("removed = %d, want 1", removed)
	}
	if _, ok := tracker.Get("old-done"); ok {
		t.Fatal("expected expired done task to be pruned")
	}
	if _, ok := tracker.Get("old-running"); !ok {
		t.Fatal("expected running task to be retained")
	}
}

func TestTrackerPrunesOldestTerminalTasksWhenOverCapacity(t *testing.T) {
	now := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	tracker := NewTrackerWithOptions(TrackerOptions{
		MaxTasks:  2,
		RetainFor: time.Hour,
		Now: func() time.Time {
			now = now.Add(time.Second)
			return now
		},
	})

	tracker.Create(TrackCreateRequest{ID: "old-done"})
	tracker.Complete("old-done", nil)
	tracker.Create(TrackCreateRequest{ID: "running"})
	tracker.Start("running")
	tracker.Create(TrackCreateRequest{ID: "new-done"})
	tracker.Complete("new-done", nil)

	if _, ok := tracker.Get("old-done"); ok {
		t.Fatal("expected oldest terminal task to be pruned")
	}
	if _, ok := tracker.Get("running"); !ok {
		t.Fatal("expected running task to be retained")
	}
	if _, ok := tracker.Get("new-done"); !ok {
		t.Fatal("expected newest terminal task to be retained")
	}
}

func TestTrackerListsTasksWithFiltersAndLimit(t *testing.T) {
	now := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	tracker := NewTrackerWithOptions(TrackerOptions{
		Now: func() time.Time {
			now = now.Add(time.Second)
			return now
		},
	})

	tracker.Create(TrackCreateRequest{ID: "u1-old", UserID: "u1", SessionID: "s1"})
	tracker.Start("u1-old")
	tracker.Create(TrackCreateRequest{ID: "u2-task", UserID: "u2", SessionID: "s2"})
	tracker.Start("u2-task")
	tracker.Create(TrackCreateRequest{ID: "u1-new", UserID: "u1", SessionID: "s1"})
	tracker.Fail("u1-new", "failed")

	got := tracker.List(TrackListRequest{UserID: "u1", Limit: 10})
	if len(got) != 2 {
		t.Fatalf("listed tasks = %#v, want 2", got)
	}
	if got[0].ID != "u1-new" || got[1].ID != "u1-old" {
		t.Fatalf("tasks not sorted newest first: %#v", got)
	}

	got = tracker.List(TrackListRequest{SessionID: "s1", Status: StatusFailed, Limit: 10})
	if len(got) != 1 || got[0].ID != "u1-new" {
		t.Fatalf("filtered tasks = %#v, want failed session task", got)
	}

	got = tracker.List(TrackListRequest{Limit: 1})
	if len(got) != 1 {
		t.Fatalf("limit result = %#v, want 1 task", got)
	}
}

func TestTrackerListReturnsCopies(t *testing.T) {
	tracker := NewTracker()
	tracker.Create(TrackCreateRequest{ID: "task-1", Steps: []string{"download_audio"}})

	got := tracker.List(TrackListRequest{Limit: 1})
	if len(got) != 1 {
		t.Fatalf("listed tasks = %#v, want 1 task", got)
	}
	got[0].Steps[0].Status = StatusFailed

	again, ok := tracker.Get("task-1")
	if !ok {
		t.Fatal("expected task to be queryable")
	}
	if again.Steps[0].Status != StatusPending {
		t.Fatalf("tracker leaked mutable listed steps: %#v", again.Steps)
	}
}

func TestNormalizeTaskListLimit(t *testing.T) {
	if got := normalizeTaskListLimit(0); got != defaultTaskListLimit {
		t.Fatalf("default limit = %d, want %d", got, defaultTaskListLimit)
	}
	if got := normalizeTaskListLimit(maxTaskListLimit + 1); got != maxTaskListLimit {
		t.Fatalf("max limit = %d, want %d", got, maxTaskListLimit)
	}
	if got := normalizeTaskListLimit(7); got != 7 {
		t.Fatalf("limit = %d, want 7", got)
	}
}
