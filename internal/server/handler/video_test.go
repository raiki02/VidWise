package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/raiki02/vidwise/internal/background"
	taskpkg "github.com/raiki02/vidwise/internal/task"
	"github.com/raiki02/vidwise/internal/tool"
)

func TestVideoProcessReturnsRequestTraceID(t *testing.T) {
	gin.SetMode(gin.TestMode)
	runner := background.NewRunner(time.Second)
	h := NewVideoHandlerWithBackground(tool.NewRegistry(), runner)
	h.process = func(context.Context, *tool.Registry, string, string, string, string, string, string, string) (string, error) {
		return "formatted text", nil
	}

	router := gin.New()
	router.Use(testTraceIDMiddleware("trace-video-1"))
	router.POST("/video/process", h.VideoProcess)

	body := bytes.NewBufferString(`{"url":"https://example.com/video","name":"demo","user_id":"u1"}`)
	req := httptest.NewRequest(http.MethodPost, "/video/process", body)
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusAccepted {
		t.Fatalf("expected status 202, got %d: %s", resp.Code, resp.Body.String())
	}

	var out VideoProcessResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out.TraceID != "trace-video-1" {
		t.Fatalf("trace id = %q, want trace-video-1", out.TraceID)
	}
	if out.TaskID == "" {
		t.Fatalf("expected task id, got %#v", out)
	}
	waitCtx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := runner.Wait(waitCtx); err != nil {
		t.Fatalf("wait: %v", err)
	}
	task, ok := h.tasks.Get(out.TaskID)
	if !ok {
		t.Fatalf("expected task %s to be tracked", out.TaskID)
	}
	if task.Status != taskpkg.StatusDone {
		t.Fatalf("task status = %q, want done", task.Status)
	}
	if task.Output["text_length"] != len("formatted text") {
		t.Fatalf("task output = %#v, want text_length", task.Output)
	}
}

func TestVideoProcessUsesDetachedBackgroundContext(t *testing.T) {
	gin.SetMode(gin.TestMode)
	runner := background.NewRunner(time.Second)
	started := make(chan struct{})
	check := make(chan struct{})
	errs := make(chan error, 1)

	h := NewVideoHandlerWithBackground(tool.NewRegistry(), runner)
	h.process = func(ctx context.Context, _ *tool.Registry, _, _, _, _, _, _, _ string) (string, error) {
		close(started)
		<-check
		errs <- ctx.Err()
		return "", nil
	}

	router := gin.New()
	router.Use(testTraceIDMiddleware("trace-video-1"))
	router.POST("/video/process", h.VideoProcess)

	reqCtx, cancel := context.WithCancel(context.Background())
	body := bytes.NewBufferString(`{"url":"https://example.com/video","name":"demo","user_id":"u1"}`)
	req := httptest.NewRequest(http.MethodPost, "/video/process", body).WithContext(reqCtx)
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusAccepted {
		t.Fatalf("expected status 202, got %d: %s", resp.Code, resp.Body.String())
	}

	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("background video process did not start")
	}
	cancel()
	close(check)

	waitCtx, waitCancel := context.WithTimeout(context.Background(), time.Second)
	defer waitCancel()
	if err := runner.Wait(waitCtx); err != nil {
		t.Fatalf("wait: %v", err)
	}
	if err := <-errs; err != nil {
		t.Fatalf("background context was canceled with request: %v", err)
	}
}

func TestVideoProcessMarksTaskFailedWhenPipelineFails(t *testing.T) {
	gin.SetMode(gin.TestMode)
	runner := background.NewRunner(time.Second)
	h := NewVideoHandlerWithBackground(tool.NewRegistry(), runner)
	h.process = func(context.Context, *tool.Registry, string, string, string, string, string, string, string) (string, error) {
		return "", errors.New("pipeline failed")
	}

	router := gin.New()
	router.Use(testTraceIDMiddleware("trace-video-1"))
	router.POST("/video/process", h.VideoProcess)

	body := bytes.NewBufferString(`{"url":"https://example.com/video","name":"demo","user_id":"u1"}`)
	req := httptest.NewRequest(http.MethodPost, "/video/process", body)
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusAccepted {
		t.Fatalf("expected status 202, got %d: %s", resp.Code, resp.Body.String())
	}

	var out VideoProcessResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}

	waitCtx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := runner.Wait(waitCtx); err != nil {
		t.Fatalf("wait: %v", err)
	}
	task, ok := h.tasks.Get(out.TaskID)
	if !ok {
		t.Fatalf("expected task %s to be tracked", out.TaskID)
	}
	if task.Status != taskpkg.StatusFailed {
		t.Fatalf("task status = %q, want failed", task.Status)
	}
	if task.Error != "pipeline failed" {
		t.Fatalf("task error = %q, want pipeline failed", task.Error)
	}
}
