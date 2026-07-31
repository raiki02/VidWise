package task

import (
	"encoding/json"
	"testing"
	"time"
)

func TestTrackedTaskRecordRoundTripKeepsTranscriptOutput(t *testing.T) {
	createdAt := time.Date(2026, 7, 31, 10, 0, 0, 0, time.UTC)
	updatedAt := createdAt.Add(time.Minute)
	startedAt := createdAt.Add(10 * time.Second)
	finishedAt := createdAt.Add(90 * time.Second)

	tracked := TrackedTask{
		ID:         "task-1",
		Type:       "video_process",
		Status:     StatusDone,
		UserID:     "user-1",
		SessionID:  "session-1",
		TraceID:    "trace-1",
		CreatedAt:  createdAt,
		UpdatedAt:  updatedAt,
		StartedAt:  &startedAt,
		FinishedAt: &finishedAt,
		Output: map[string]any{
			"text":                "formatted transcript",
			"raw_text":            "raw transcript",
			"formatted_text":      "formatted transcript",
			"text_formatted":      true,
			"can_index_knowledge": true,
		},
	}

	record, err := taskRecordFromTracked(tracked)
	if err != nil {
		t.Fatalf("taskRecordFromTracked returned error: %v", err)
	}
	if record.ID != tracked.ID || record.UserID != tracked.UserID || record.SessionID != tracked.SessionID {
		t.Fatalf("record identity = %#v, want tracked identity", record)
	}
	if record.TraceID == nil || *record.TraceID != "trace-1" {
		t.Fatalf("TraceID = %#v, want trace-1", record.TraceID)
	}
	if record.StartedAt == nil || !record.StartedAt.Equal(startedAt) || record.FinishedAt == nil || !record.FinishedAt.Equal(finishedAt) {
		t.Fatalf("record timing = started %#v finished %#v", record.StartedAt, record.FinishedAt)
	}

	var output map[string]any
	if err := json.Unmarshal(record.Output, &output); err != nil {
		t.Fatalf("decode output: %v", err)
	}
	if output["formatted_text"] != "formatted transcript" || output["raw_text"] != "raw transcript" {
		t.Fatalf("record output = %#v, want transcript fields", output)
	}

	restored := trackedTaskFromRecord(record, []Step{{
		TaskID:    "task-1",
		Name:      "transcribe_audio",
		Status:    StatusDone,
		CreatedAt: createdAt,
		UpdatedAt: updatedAt,
	}}, Transcript{})
	if restored.Output["formatted_text"] != "formatted transcript" || restored.Output["raw_text"] != "raw transcript" {
		t.Fatalf("restored output = %#v, want transcript fields", restored.Output)
	}
	if len(restored.Steps) != 1 || restored.Steps[0].Name != "transcribe_audio" || restored.Steps[0].Status != StatusDone {
		t.Fatalf("restored steps = %#v", restored.Steps)
	}
}

func TestTrackedTaskFromRecordUsesTranscriptWhenOutputMissing(t *testing.T) {
	createdAt := time.Date(2026, 7, 31, 10, 0, 0, 0, time.UTC)
	rawText := "raw transcript"
	formattedText := "formatted transcript"

	restored := trackedTaskFromRecord(Task{
		ID:        "task-1",
		Type:      "video_process",
		Status:    StatusDone,
		UserID:    "user-1",
		SessionID: "session-1",
		CreatedAt: createdAt,
		UpdatedAt: createdAt,
	}, nil, Transcript{
		TaskID:        "task-1",
		SessionID:     "session-1",
		UserID:        "user-1",
		RawText:       &rawText,
		FormattedText: &formattedText,
	})

	if restored.Output["text"] != formattedText {
		t.Fatalf("text = %#v, want formatted transcript", restored.Output["text"])
	}
	if restored.Output["raw_text"] != rawText || restored.Output["formatted_text"] != formattedText {
		t.Fatalf("restored output = %#v, want transcript fields", restored.Output)
	}
	if restored.Output["text_formatted"] != true || restored.Output["can_index_knowledge"] != true {
		t.Fatalf("restored output = %#v, want indexable formatted transcript", restored.Output)
	}
}
