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

func TestListTasksReturnsScopedTasks(t *testing.T) {
	gin.SetMode(gin.TestMode)
	tracker := taskpkg.NewTracker()
	tracker.Create(taskpkg.TrackCreateRequest{ID: "u1-running", UserID: "u1", SessionID: "s1"})
	tracker.Start("u1-running")
	tracker.Create(taskpkg.TrackCreateRequest{ID: "u1-failed", UserID: "u1", SessionID: "s1"})
	tracker.Fail("u1-failed", "failed")
	tracker.Create(taskpkg.TrackCreateRequest{ID: "u2-running", UserID: "u2", SessionID: "s2"})
	tracker.Start("u2-running")

	h := NewTaskHandlerWithTracker(tracker)
	router := gin.New()
	router.GET("/tasks", h.ListTasks)

	req := httptest.NewRequest(http.MethodGet, "/tasks?user_id=u1&status=failed&limit=10", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
	}

	var out TaskListResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out.Count != 1 || len(out.Tasks) != 1 {
		t.Fatalf("unexpected task list: %#v", out)
	}
	if out.Tasks[0].ID != "u1-failed" {
		t.Fatalf("task id = %q, want u1-failed", out.Tasks[0].ID)
	}
}

func TestListTasksReadsScopeFromHeaders(t *testing.T) {
	gin.SetMode(gin.TestMode)
	tracker := taskpkg.NewTracker()
	tracker.Create(taskpkg.TrackCreateRequest{ID: "session-task", UserID: "u1", SessionID: "header-session"})

	h := NewTaskHandlerWithTracker(tracker)
	router := gin.New()
	router.GET("/tasks", h.ListTasks)

	req := httptest.NewRequest(http.MethodGet, "/tasks", nil)
	req.Header.Set("X-Session-ID", " header-session ")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
	}

	var out TaskListResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out.Count != 1 || out.Tasks[0].ID != "session-task" {
		t.Fatalf("unexpected task list: %#v", out)
	}
}

func TestListTasksRequiresScope(t *testing.T) {
	gin.SetMode(gin.TestMode)
	h := NewTaskHandlerWithTracker(taskpkg.NewTracker())
	router := gin.New()
	router.Use(testTraceIDMiddleware("trace-task-1"))
	router.GET("/tasks", h.ListTasks)

	req := httptest.NewRequest(http.MethodGet, "/tasks", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d: %s", resp.Code, resp.Body.String())
	}

	var out map[string]any
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out["error"] != "user_id or session_id is required" {
		t.Fatalf("unexpected error response: %#v", out)
	}
	if out["trace_id"] != "trace-task-1" {
		t.Fatalf("trace_id = %#v, want trace-task-1", out["trace_id"])
	}
}

func TestListTasksRejectsInvalidFilters(t *testing.T) {
	gin.SetMode(gin.TestMode)
	h := NewTaskHandlerWithTracker(taskpkg.NewTracker())
	router := gin.New()
	router.GET("/tasks", h.ListTasks)

	tests := []string{
		"/tasks?user_id=u1&status=unknown",
		"/tasks?user_id=u1&limit=0",
	}
	for _, path := range tests {
		t.Run(path, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, path, nil)
			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			if resp.Code != http.StatusBadRequest {
				t.Fatalf("expected status 400, got %d: %s", resp.Code, resp.Body.String())
			}
		})
	}
}
