package server

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestTraceIDPropagatesHeaderAndRequestLoggerEmitsStructuredLog(t *testing.T) {
	gin.SetMode(gin.TestMode)
	var logs bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&logs, &slog.HandlerOptions{}))

	engine := gin.New()
	engine.Use(TraceID())
	engine.Use(RequestLoggerWithLogger(logger))
	engine.GET("/ping", func(c *gin.Context) {
		c.JSON(http.StatusAccepted, gin.H{"ok": true})
	})

	req := httptest.NewRequest(http.MethodGet, "/ping", nil)
	req.Header.Set(traceIDHeader, "trace-123")
	resp := httptest.NewRecorder()

	engine.ServeHTTP(resp, req)

	if resp.Code != http.StatusAccepted {
		t.Fatalf("status = %d, want %d", resp.Code, http.StatusAccepted)
	}
	if got := resp.Header().Get(traceIDHeader); got != "trace-123" {
		t.Fatalf("trace header = %q, want trace-123", got)
	}
	if resp.Header().Get("X-Response-Time") == "" {
		t.Fatalf("expected response time header")
	}

	var entry map[string]any
	if err := json.Unmarshal(bytes.TrimSpace(logs.Bytes()), &entry); err != nil {
		t.Fatalf("decode log entry %q: %v", logs.String(), err)
	}
	if entry["msg"] != "http.request" {
		t.Fatalf("log msg = %#v, want http.request", entry["msg"])
	}
	if entry["trace_id"] != "trace-123" {
		t.Fatalf("trace_id = %#v, want trace-123", entry["trace_id"])
	}
	if entry["method"] != http.MethodGet || entry["path"] != "/ping" {
		t.Fatalf("unexpected request fields: %#v", entry)
	}
	if entry["status"] != float64(http.StatusAccepted) {
		t.Fatalf("status field = %#v, want %d", entry["status"], http.StatusAccepted)
	}
	if _, ok := entry["latency_ms"]; !ok {
		t.Fatalf("expected latency_ms field, got %#v", entry)
	}
}
