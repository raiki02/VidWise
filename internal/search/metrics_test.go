package search

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"testing"
	"time"

	"github.com/raiki02/vidwise/internal/trace"
)

func TestLoggingMetricsIncludesTraceID(t *testing.T) {
	var logs bytes.Buffer
	metrics := NewLoggingMetrics(slog.New(slog.NewJSONHandler(&logs, &slog.HandlerOptions{})))

	metrics.ObserveSearch(trace.WithID(t.Context(), "trace-search-1"), "ok", 12*time.Millisecond)

	var entry map[string]any
	if err := json.Unmarshal(bytes.TrimSpace(logs.Bytes()), &entry); err != nil {
		t.Fatalf("decode log entry %q: %v", logs.String(), err)
	}
	if entry["msg"] != "search.metric.search" {
		t.Fatalf("msg = %#v, want search.metric.search", entry["msg"])
	}
	if entry["trace_id"] != "trace-search-1" {
		t.Fatalf("trace_id = %#v, want trace-search-1", entry["trace_id"])
	}
	if entry["status"] != "ok" {
		t.Fatalf("status = %#v, want ok", entry["status"])
	}
	if entry["elapsed_ms"] != float64(12) {
		t.Fatalf("elapsed_ms = %#v, want 12", entry["elapsed_ms"])
	}
}
