package handler

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/raiki02/vidwise/internal/tool"
)

func TestVideoProcessReturnsRequestTraceID(t *testing.T) {
	gin.SetMode(gin.TestMode)
	h := NewVideoHandler(tool.NewRegistry())

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
}
