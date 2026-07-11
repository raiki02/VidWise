package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	einotool "github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/schema"
	"github.com/gin-gonic/gin"
	"github.com/raiki02/vidwise/internal/agent"
	"github.com/raiki02/vidwise/internal/background"
	taskpkg "github.com/raiki02/vidwise/internal/task"
	"github.com/raiki02/vidwise/internal/tool"
)

func TestVideoProcessReturnsRequestTraceID(t *testing.T) {
	gin.SetMode(gin.TestMode)
	runner := background.NewRunner(time.Second)
	h := NewVideoHandlerWithBackground(tool.NewRegistry(), runner)
	h.processWithObserver = func(_ context.Context, _ *tool.Registry, _, _, _, _, _, _, _ string, observer agent.VideoProcessObserver) (string, error) {
		observer.StepStarted(agent.VideoProcessStepDownloadAudio)
		observer.StepDone(agent.VideoProcessStepDownloadAudio)
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
	if task.Output["text"] != "formatted text" {
		t.Fatalf("task output = %#v, want transcript text", task.Output)
	}
	if task.Output["knowledge_indexed"] != false {
		t.Fatalf("task output = %#v, want knowledge_indexed=false", task.Output)
	}
	if len(task.Steps) != len(agent.VideoProcessStepNames()) {
		t.Fatalf("task steps = %#v", task.Steps)
	}
	if task.Steps[0].Name != agent.VideoProcessStepDownloadAudio || task.Steps[0].Status != taskpkg.StatusDone {
		t.Fatalf("download step = %#v, want done", task.Steps[0])
	}
}

func TestVideoProcessUsesDetachedBackgroundContext(t *testing.T) {
	gin.SetMode(gin.TestMode)
	runner := background.NewRunner(time.Second)
	started := make(chan struct{})
	check := make(chan struct{})
	errs := make(chan error, 1)

	h := NewVideoHandlerWithBackground(tool.NewRegistry(), runner)
	h.processWithObserver = func(ctx context.Context, _ *tool.Registry, _, _, _, _, _, _, _ string, _ agent.VideoProcessObserver) (string, error) {
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
	h.processWithObserver = func(_ context.Context, _ *tool.Registry, _, _, _, _, _, _, _ string, observer agent.VideoProcessObserver) (string, error) {
		observer.StepStarted(agent.VideoProcessStepTranscribeAudio)
		observer.StepFailed(agent.VideoProcessStepTranscribeAudio, errors.New("pipeline failed"))
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
	if task.Steps[1].Status != taskpkg.StatusFailed {
		t.Fatalf("transcribe step = %#v, want failed", task.Steps[1])
	}
}

func TestIndexTaskTranscriptIndexesCompletedTaskText(t *testing.T) {
	gin.SetMode(gin.TestMode)
	registry := tool.NewRegistry()
	rag := &recordingRAGIndexTool{output: `{"chunk_count":2,"content_type":"text/plain","source_ids":["source-1"]}`}
	registry.Register("rag_index", rag, nil)
	tracker := taskpkg.NewTracker()
	tracker.Create(taskpkg.TrackCreateRequest{
		ID:        "task-1",
		Type:      "video_process",
		UserID:    "u1",
		SessionID: "s1",
	})
	tracker.Complete("task-1", map[string]any{
		"text":              "formatted transcript",
		"text_length":       len("formatted transcript"),
		"filename":          "demo.txt",
		"source_url":        "https://example.com/video",
		"knowledge_indexed": false,
	})
	h := NewVideoHandlerWithBackgroundAndTasks(registry, background.NewRunner(time.Second), tracker)

	router := gin.New()
	router.POST("/task/:id/index", h.IndexTaskTranscript)

	req := httptest.NewRequest(http.MethodPost, "/task/task-1/index", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
	}
	if !strings.Contains(rag.input, `"text":"formatted transcript"`) {
		t.Fatalf("rag input missing transcript text: %s", rag.input)
	}
	if !strings.Contains(rag.input, `"task_id":"task-1"`) {
		t.Fatalf("rag input missing task metadata: %s", rag.input)
	}

	var out VideoTaskIndexResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out.Status != "indexed" || out.ChunkCount != 2 || len(out.SourceIDs) != 1 || out.SourceIDs[0] != "source-1" {
		t.Fatalf("unexpected index response: %#v", out)
	}
	task, ok := tracker.Get("task-1")
	if !ok {
		t.Fatal("expected task")
	}
	if task.Output["knowledge_indexed"] != true {
		t.Fatalf("task output = %#v, want indexed", task.Output)
	}
}

func TestIndexTaskTranscriptRejectsRunningTask(t *testing.T) {
	gin.SetMode(gin.TestMode)
	tracker := taskpkg.NewTracker()
	tracker.Create(taskpkg.TrackCreateRequest{ID: "task-1", Type: "video_process", UserID: "u1"})
	tracker.Start("task-1")
	h := NewVideoHandlerWithBackgroundAndTasks(tool.NewRegistry(), background.NewRunner(time.Second), tracker)

	router := gin.New()
	router.POST("/task/:id/index", h.IndexTaskTranscript)

	req := httptest.NewRequest(http.MethodPost, "/task/task-1/index", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusConflict {
		t.Fatalf("expected status 409, got %d: %s", resp.Code, resp.Body.String())
	}
}

type recordingRAGIndexTool struct {
	input  string
	output string
	err    error
}

func (t *recordingRAGIndexTool) Info(context.Context) (*schema.ToolInfo, error) {
	return &schema.ToolInfo{Name: "rag_index"}, nil
}

func (t *recordingRAGIndexTool) InvokableRun(_ context.Context, input string, _ ...einotool.Option) (string, error) {
	t.input = input
	if t.err != nil {
		return "", t.err
	}
	return t.output, nil
}
