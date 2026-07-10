package handler

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	taskpkg "github.com/raiki02/vidwise/internal/task"
)

func TestGetTaskReturnsTrackedTask(t *testing.T) {
	gin.SetMode(gin.TestMode)
	tracker := taskpkg.NewTracker()
	tracker.Create(taskpkg.TrackCreateRequest{
		ID:        "task-1",
		Type:      "video_process",
		UserID:    "u1",
		SessionID: "s1",
		TraceID:   "trace-1",
	})
	tracker.Start("task-1")

	h := NewTaskHandlerWithTracker(tracker)
	router := gin.New()
	router.GET("/task/:id", h.GetTask)

	req := httptest.NewRequest(http.MethodGet, "/task/task-1", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
	}

	var out taskpkg.TrackedTask
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out.ID != "task-1" || out.Status != taskpkg.StatusRunning {
		t.Fatalf("unexpected task response: %#v", out)
	}
	if out.TraceID != "trace-1" {
		t.Fatalf("trace id = %q, want trace-1", out.TraceID)
	}
}

func TestGetTaskReturnsNotFoundForMissingTask(t *testing.T) {
	gin.SetMode(gin.TestMode)
	h := NewTaskHandlerWithTracker(taskpkg.NewTracker())
	router := gin.New()
	router.Use(testTraceIDMiddleware("trace-task-1"))
	router.GET("/task/:id", h.GetTask)

	req := httptest.NewRequest(http.MethodGet, "/task/missing", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusNotFound {
		t.Fatalf("expected status 404, got %d: %s", resp.Code, resp.Body.String())
	}

	var out map[string]any
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out["error"] != "task not found" {
		t.Fatalf("unexpected error response: %#v", out)
	}
	if out["trace_id"] != "trace-task-1" {
		t.Fatalf("trace_id = %#v, want trace-task-1", out["trace_id"])
	}
}
